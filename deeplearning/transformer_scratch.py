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
