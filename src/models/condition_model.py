import einops
import math
import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast
from torchinfo import summary
from torchvision.models.vision_transformer import MLPBlock
import torch.nn.functional as F


class ConditionalAttentionEncoder(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, condition: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.cross_attention(query=condition, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class ConditionalQuery(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_attr: int,
            seq_length: int,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.embedding = nn.Embedding(num_attr, hidden_dim)

    def forward(self, attr_idx):
        x = self.embedding(attr_idx)
        return einops.repeat(x, 'b d -> b n d', n=self.seq_length)


class ConditionCA(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """

    def __init__(self, attr_nums, backbone='vit', mode='singleton'):
        super(ConditionCA, self).__init__()
        self.mode = mode
        self.attr_nums = attr_nums
        self.backbone = torchvision.models.vit_b_16()
        self.backbone.load_state_dict(torch.hub.load_state_dict_from_url(
            url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
            map_location=torch.device('cpu')
        ))
        self.backbone.encoder.ln = nn.Sequential()
        self.backbone.heads = nn.Sequential()
        self.hidden_dim = self.backbone.hidden_dim
        self.dim_chunk = 340

        self.con_q = ConditionalQuery(self.hidden_dim, len(attr_nums), self.backbone.seq_length)
        self.con_encoder = ConditionalAttentionEncoder(
            num_heads=12,
            hidden_dim=self.hidden_dim,
            mlp_dim=3072,
            dropout=0.1,
            attention_dropout=0.1
        )

        self.ln = nn.LayerNorm(self.hidden_dim, eps=1e-6)

        if mode == 'branch':
            dis_proj = []
            for i in range(len(attr_nums)):
                dis_proj.append(
                    nn.Linear(self.hidden_dim, self.dim_chunk),
                )
            self.dis_proj = nn.ModuleList(dis_proj)
        else:
            self.dis_proj = nn.Linear(self.hidden_dim, self.dim_chunk)

        attr_classifier = []
        for i in range(len(attr_nums)):
            attr_classifier.append(
                nn.Linear(self.dim_chunk, attr_nums[i])
            )
        self.attr_classifier = nn.ModuleList(attr_classifier)

    @autocast()
    def forward(self, img):
        """
        Returns:
            attr_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        x = self.backbone._process_input(img)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)  # (batch, 196+1, 768)

        attr_cls_token = []
        for i in range(len(self.attr_nums)):
            i = torch.tensor(i, device=img.device)
            i = i.repeat(img.shape[0])
            condition = self.con_q(i)
            attr_token = self.ln(self.con_encoder(x, condition))
            attr_cls_token.append(attr_token[:, 0])

        attr_feat = []
        if self.mode == 'branch':
            for i, layer in enumerate(self.dis_proj):
                attr_feat.append(layer(attr_cls_token[i]))
        else:
            for i in range(len(attr_cls_token)):
                attr_feat.append(self.dis_proj(attr_cls_token[i]))
        attr_classification_out = []
        for i, layer in enumerate(self.attr_classifier):
            attr_classification_out.append(layer(attr_feat[i]))
        return attr_feat, attr_classification_out


class ManipulateBlock(nn.Module):
    """
    输入需要修改的特征, 以及其操作指示, 通过交叉注意力使其变为目标属性值得到特征,
    通过
    """

    def __init__(self, attr_num, hidden_dim, dim_chunk):
        super().__init__()
        self.indicator_embed = nn.Sequential(
            nn.Linear(sum(attr_num), hidden_dim),
            nn.Linear(hidden_dim, dim_chunk * len(attr_num)),
            nn.ReLU()
        )
        self.attr_num = attr_num
        self.ln = nn.LayerNorm(dim_chunk, eps=1e-6)
        # 默认num_heads为10
        self.manipulate = nn.MultiheadAttention(dim_chunk, 10, dropout=0.1, batch_first=True)

    @autocast()
    def forward(self, feat, indicator):
        b = feat.shape[0]
        q = self.indicator_embed(indicator)  # (b,340)
        q = einops.repeat(q, 'b (l d) -> b l d', b=b, l=len(self.attr_num))  # (b, 12, 340)
        kv = einops.rearrange(feat, 'b (l d) -> b l d', b=b, l=len(self.attr_num))
        target, _ = self.manipulate(q, kv, kv, need_weights=False)
        target = self.ln(target)

        return einops.rearrange(target, 'b l d -> b (l d)')


class ManipulateBlockV2(nn.Module):

    def __init__(self, attr_num, hidden_dim, dim_chunk):
        super().__init__()
        self.indicator_embed = nn.Sequential(
            nn.Linear(sum(attr_num), hidden_dim),
            nn.Linear(hidden_dim, dim_chunk * len(attr_num)),
            nn.ReLU()
        )
        self.manip_queries = nn.Parameter(torch.empty(len(attr_num), dim_chunk))
        nn.init.kaiming_uniform_(self.manip_queries, a=math.sqrt(5))
        self.attention_pooler = nn.MultiheadAttention(dim_chunk, 10, dropout=0.1, kdim=dim_chunk,
                                                      vdim=dim_chunk, bias=False, batch_first=True)
        self.ln1 = nn.LayerNorm(dim_chunk, eps=1e-6)
        self.ln2 = nn.LayerNorm(dim_chunk, eps=1e-6)
        self.manipulate = nn.MultiheadAttention(dim_chunk, 10, dropout=0.1, batch_first=True)
        self.attr_num = attr_num

    @autocast()
    def forward(self, feat, indicator):
        b = feat.shape[0]
        q = self.indicator_embed(indicator)  # (b,340)
        q = einops.repeat(q, 'b (l d) -> b l d', b=b, l=len(self.attr_num))  # (b, 12, 340)
        manip_queries = einops.repeat(self.manip_queries, 'l d -> b l d', b=b, l=len(self.attr_num))
        manip_queries = self.ln1(manip_queries + q)
        feat = einops.rearrange(feat, 'b (l d) -> b l d', b=b, l=len(self.attr_num))
        manip_token, _ = self.attention_pooler(query=manip_queries, key=feat, value=feat, need_weights=False)
        manip_token = self.ln2(manip_token)

        target, _ = self.manipulate(query=manip_token, key=feat, value=feat, need_weights=False)

        return einops.rearrange(target, 'b l d -> b (l d)'), einops.rearrange(manip_token, 'b l d -> b (l d)')


class ManipulateBlockAAC(nn.Module):
    def __init__(self, attr_num, hidden_dim, dim_chunk, num_heads=10, dropout=0.1):
        super().__init__()
        self.attr_num = attr_num
        self.hidden_dim = hidden_dim
        self.dim_chunk = dim_chunk
        self.num_heads = num_heads

        # 共享的 indicator_embed 层
        self.indicator_embed = nn.Sequential(
            nn.Linear(sum(attr_num), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_chunk * len(attr_num)),
            nn.Dropout(dropout),
        )

        # LayerNorm
        self.ln = nn.LayerNorm(dim_chunk, eps=1e-6)

        # fc1 和 fc2 共享，用于每个头的处理
        self.fc1 = nn.Sequential(
            nn.Linear(dim_chunk, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_chunk),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(dim_chunk, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_chunk),
            nn.Dropout(dropout)
        )


    @autocast()
    def forward(self, feat, indicator):
        b = feat.shape[0]

        # 共享的 indicator_embed 用于计算 q
        q = self.indicator_embed(indicator)  # (b, 340)
        q = einops.repeat(q, 'b (l d) -> b l d', b=b, l=len(self.attr_num))  # (b, 12, 340)

        # fc1 输出 h
        h = einops.rearrange(feat, 'b (l d) -> b l d', b=b, l=len(self.attr_num)) # (b, 12, 340)
        h = self.fc1(h) # (b, 12, 340)

        # 拆分 q 和 h 在 dim_chunk 维度上
        q = q.view(b, len(self.attr_num), self.num_heads, self.dim_chunk // self.num_heads)  # (b, 12, num_heads, dim_chunk // num_heads)
        h = h.view(b, len(self.attr_num), self.num_heads, self.dim_chunk // self.num_heads)  # (b, 12, num_heads, dim_chunk // num_heads)

        # 计算点积注意力：注意 q 和 h 是按头拆分的
        atten_output = torch.einsum('blhd,blhd->blh', q, h)  # (b, 12, num_heads)
        scaled_atten_output = atten_output / torch.sqrt(
            torch.tensor(self.dim_chunk // self.num_heads, dtype=atten_output.dtype, device=atten_output.device))
        atten_weights = F.softmax(scaled_atten_output, dim=1)  # (b, 12, num_heads)

        v_head = atten_weights.unsqueeze(-1) * h  # (batch, 12, num_heads, dim_chunk // num_heads)
        # 累加 v_head 获取当前head下的上下文向量
        context = torch.sum(v_head, dim=1, keepdim=True)  # (batch, 1, num_heads, dim_chunk // num_heads)
        # 将上下文向量与 h_head 按元素相乘
        context_scaled_heads = context * h  # (batch, 12, num_heads, dim_chunk // num_heads)

        # 将所有头的输出拼接
        stacked_v = context_scaled_heads.view(b, len(self.attr_num), self.dim_chunk)  # (b, 12, dim_chunk)

        # 最终投影以及残差
        output = self.fc2(stacked_v)
        output = output + h.view(b, len(self.attr_num), self.dim_chunk)

        # 层归一化
        output = self.ln(output)

        # 恢复为 (b, l * dim_chunk) 的形状
        return einops.rearrange(output, 'b l d -> b (l d)')


if __name__ == '__main__':
    attr_nums = [7, 3, 3, 4, 6, 3]
    feat = torch.randn((2, 6 * 340), device='cpu')
    indi = torch.Tensor([[-1, 1, 0, 0, 0, 0, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 0],
                         [-1, 1, 0, 0, 0, 0, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 0]],
                        device='cpu')
    manip = ManipulateBlockV2(attr_nums, 768, 340).to('cpu')
    target = manip(feat, indi)
    summary(manip, input_data=(feat, indi), depth=5, col_names=['input_size', 'output_size', 'params_percent'], )
