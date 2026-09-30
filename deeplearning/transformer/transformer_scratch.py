import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
print("Pytorch version:", torch.__version__)
print("计算设备:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class Embedding(nn.Module):
    """
    手写实现词嵌入层
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 可学习权重：每行是一个token的向量
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, x):
        """
        前向传播
        ids: (B, L)整数 -> (B, L, d_model),weight[ids]就是按行查表
        """
        # (B, L) -> (B, L, d_model)
        return self.weight[x]


"""
实质上就是词表索引的位序
emb = Embedding(13, 16)
ref = nn.Embedding(13, 16)
with torch.no_grad():
    ref.weight.data.copy_(emb.weight.data)

ids = torch.randint(0, 13, (2, 5))
print("ids:\n", ids)
emb_ids = emb(ids)
ref_ids = ref(ids)
one_hot_ids = F.one_hot(ids, 13)
print("emb weight:\n", emb.weight)

print(torch.allclose(emb(ids), ref(ids)))
print(torch.allclose(emb(ids), one_hot_ids.float() @ emb.weight))
"""


class PositionalEncoding(nn.Module):
    """
    正弦-余弦位置编码
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 即10000^(2i/d_model)
        pe[:, 0::2] = torch.sin(position / div_term)  # 偶数维用sin
        pe[:, 1::2] = torch.cos(position / div_term)  # 奇数维用cos
        self.register_buffer('pe', pe.unsqueeze(0))  # （1,max_len,d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


pe_mod = PositionalEncoding(16, max_len=64, dropout=0.0)

print("PE 位置0：sin=0，cos=1",
      torch.allclose(pe_mod.pe[0, 0, 0::2], torch.zeros(8), atol=1e-6) and torch.allclose(pe_mod.pe[0, 1, 1::2],
                                                                                          torch.ones(8), atol=1e-6))

print("PE 值域：", (pe_mod.pe.abs() <= 1.0).all().item())
x_tmp = torch.randn(2, 10, 16)
print("PE forward shape:", pe_mod(x_tmp).shape == (2, 10, 16))
