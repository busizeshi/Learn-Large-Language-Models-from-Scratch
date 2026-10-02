# =====================================================================
# 哈佛《The Annotated Transformer》源码：嵌入层（Embeddings）
# =====================================================================
import math
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """把 token ID 序列映射为 d_model 维的稠密向量序列。

    参数：
        d_model : 模型隐藏维度（论文中为 512），决定每个 token 被映射成多长的向量
        vocab   : 词表大小，即模型认识的 token 总数（嵌入矩阵的行数）
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 查找表（lookup table）：本质是一个形状为 (vocab, d_model) 的可学习矩阵，
        # 词表中每个 token 独占一行，这一行就是它的"词向量"
        self.lut = nn.Embedding(vocab, d_model)
        # 记录模型维度，供 forward 中缩放使用
        self.d_model = d_model

    def forward(self, x):
        # ① 查表：把每个 token ID 替换为嵌入矩阵中对应的那一行
        #    形状变化：(batch, seq_len) --> (batch, seq_len, d_model)
        # ② 缩放：乘以 sqrt(d_model)，把嵌入的数值量级拉回与 d_model 解耦的水平（下文详细解释）
        tmp=self.lut(x)
        return self.lut(x) * math.sqrt(self.d_model)

# =====================================================================
# 实验 1：实例化嵌入层，亲眼看一次"查表"过程
# =====================================================================
torch.manual_seed(42)

# 用极小规模方便观察：词表 10 个 token，每个映射为 8 维向量
d_model, vocab_size = 8, 10
embedding = Embeddings(d_model, vocab_size)

# 一批 token ID：2 个"句子"，每句 4 个 token
token_ids = torch.tensor([
    [0, 3, 5, 9],
    [1, 2, 7, 4],
])

print(f"嵌入矩阵（查找表）形状 : {tuple(embedding.lut.weight.shape)}          -> (vocab, d_model)")
print(f"输入 token ID 形状     : {tuple(token_ids.shape)}                 -> (batch, seq_len)")

output = embedding(token_ids)
print(f"输出嵌入向量形状       : {tuple(output.shape)}              -> (batch, seq_len, d_model)")

print("-" * 62)
# token 5 的词向量就是嵌入矩阵的第 5 行
print("嵌入矩阵第 5 行（token 5 的原始词向量）:")
print(embedding.lut.weight[5])

# 输出中第 1 个句子的第 3 个位置正是 token 5（注意已乘 sqrt(d_model)）
print("\n前向传播输出 output[0, 2]（= 原始词向量 x sqrt(d_model)）:")
print(output[0, 2])

print("\n验证 output[0, 2] == weight[5] * sqrt(d_model) ->",
      torch.allclose(output[0, 2], embedding.lut.weight[5] * math.sqrt(d_model)))

# =====================================================================
# 实验 2：nn.Embedding 查表 vs 独热编码 × 矩阵
# =====================================================================
import torch.nn.functional as F

# 方式一：嵌入层的真实做法 —— 直接用 ID 索引取行
by_lookup = embedding.lut(token_ids)                            # (2, 4, 8)

# 方式二：数学本质 —— 先构造独热编码，再做矩阵乘法
one_hot = F.one_hot(token_ids, num_classes=vocab_size).float()  # (2, 4, 10)
by_matmul = one_hot @ embedding.lut.weight                      # (2,4,10) @ (10,8) -> (2, 4, 8)

print(f"独热编码形状: {tuple(one_hot.shape)}，每个 token 的 10 维向量中仅 1 个 1（90% 是 0）")
print(f"两种方式结果是否完全一致: {torch.allclose(by_lookup, by_matmul)}")

batch, seq_len = token_ids.shape
print(f"\n本例计算量: 查表 {batch * seq_len} 次索引  vs  矩阵乘法 {batch * seq_len * vocab_size} 次乘加")
print("真实场景 vocab=30000 时，矩阵乘法每个词要做 30000 次乘加，其中 99.997% 都在与 0 相乘")

# =====================================================================
# 实验 3：乘 sqrt(d_model) 前后，嵌入量级如何随 d_model 变化
# =====================================================================
print(f"{'d_model':>8} | {'未缩放 每维std':>14} | {'未缩放 向量模长':>15} | {'缩放后 每维std':>14} | {'缩放后 向量模长':>15}")
print("-" * 82)

for d in [64, 512, 2048]:
    torch.manual_seed(0)
    lut = nn.Embedding(1000, d)
    # 复现论文式小方差初始化：每维标准差 = 1/sqrt(d_model)
    nn.init.normal_(lut.weight, mean=0.0, std=d ** -0.5)

    ids = torch.randint(0, 1000, (1, 100))   # 随机抽取 100 个 token
    raw = lut(ids)[0]                        # (100, d) 未缩放
    scaled = raw * math.sqrt(d)              # 缩放后

    print(f"{d:>8} | {raw.std():>14.3f} | {raw.norm(dim=-1).mean():>16.3f} "
          f"| {scaled.std():>14.3f} | {scaled.norm(dim=-1).mean():>16.3f}")

print("\n观察：")
print("① 未缩放时每维 std = 1/sqrt(d)：d 越大信号越弱（0.125 -> 0.022），位置编码会彻底淹没词义")
print("② 缩放后每维 std 稳定在 1.0 附近 —— 嵌入量级与 d_model 彻底解耦")
print("③ 缩放后模长约为 sqrt(d)，恰好与正弦位置编码（模长约为 sqrt(d/2) ~ sqrt(d)）处于同一量级")

# =====================================================================
# 实验 4：稀疏更新 —— 只有被查过的行才收到梯度
# =====================================================================
torch.manual_seed(42)
lut = nn.Embedding(5, 3)            # 词表 5 个 token，每个 3 维
out = lut(torch.tensor([2]))        # 本轮只查询了第 2 行
out.sum().backward()

print("嵌入矩阵的梯度（逐行绝对值求和）：")
print(lut.weight.grad.abs().sum(dim=1))

print("\n解读：只有第 2 行梯度非零，其余行纹丝不动 ——")
print("嵌入层的参数更新是'按需'的，这正是查表机制在优化层面带来的稀疏性。")