# -*- coding: utf-8 -*-
"""
Llama3 分词器实现（重构自 Meta 官方源码，中文注释版）
=====================================================

官方源码: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
配套讲解: 同目录 token.ipynb 中「## Llama3分词器源码」一节

依赖: pip install tiktoken regex
运行: python llama3_tokenizer.py  （自动执行底部的演示代码）

模块组成（按数据流顺序）:
    1. my_load_tiktoken_bpe : 手写词表加载器（与官方 load_tiktoken_bpe 等价）
    2. Tokenizer            : 核心分词器，完成 文本 <-> token 编号
    3. ChatFormat           : 对话格式化器，把多轮对话拼成模型 prompt
"""
import base64
import os
import regex
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken                      # OpenAI 开源的高性能 BPE 分词库（Rust 实现）
from tiktoken.load import load_tiktoken_bpe  # 官方提供的词表读取函数

# 词表文件路径（与本文件同目录，用 __file__ 定位保证任何 cwd 下都能找到）
TOKENIZER_PATH = str(Path(__file__).resolve().parent / "tokenizer.model")


# ============================================================
# 第 1 步：加载词表
# ============================================================
def my_load_tiktoken_bpe(model_path: str) -> Dict[bytes, int]:
    """读取 tiktoken 词表文件，返回 {词元(bytes): 编号(int)} 映射。

    tiktoken 词表就是纯文本文件，每行格式为「base64编码的词元 编号」，例如::

        IQ== 0        →  b'!' 的编号是 0
        SGVsbG8= 153839  →  b'Hello' 的编号是 153839

    （Llama3 虽然保留了 .model 后缀，但内部已经不是 SentencePiece 了）
    """
    mergeable_ranks: Dict[bytes, int] = {}
    with open(model_path, "rb") as f:
        for line in f:                       # 逐行读取
            line = line.strip()
            if not line:                     # 跳过空行
                continue
            token_b64, rank = line.split()   # 拆成 [base64词元, 编号] 两列
            token = base64.b64decode(token_b64)  # base64 解码，还原词元的真实字节
            mergeable_ranks[token] = int(rank)
    return mergeable_ranks


# ============================================================
# 第 2 步：Tokenizer 类
# ============================================================
Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    """一条对话消息：角色 + 内容"""
    role: Role
    content: str

Dialog = Sequence[Message]  # 一段对话 = 若干条消息


class Tokenizer:
    """Llama3 分词器：负责「文本 <-> token 编号」的相互转换。

    整个类只做三件事：
      1. __init__  : 加载词表、登记特殊 token、创建 tiktoken 编码器
      2. encode    : 文本 → token 编号列表
      3. decode    : token 编号列表 → 文本
    """

    # ---------- 类常量 ----------
    num_reserved_special_tokens = 256   # 词表尾部预留 256 个特殊 token 槽位

    # BPE 预切分正则：先把文本"粗切"成小片段，再对每个片段做 BPE 合并。
    # 规则大致是：英文缩写('s/'t/...)、单词、1~3位数字、标点符号串、换行、其余空白
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str):
        """加载词表并初始化 tiktoken 编码器。"""
        assert os.path.isfile(model_path), model_path

        # ① 加载普通词表：{b'!': 0, b'Hello': 153839, ...}，共 128,000 项
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        # ② 登记特殊 token：占用 id 128000 ~ 128255，共 256 个
        #    前 10 个有固定用途，其余 246 个是纯预留（微调时可以自定义）
        special_tokens = [
            "<|begin_of_text|>",              # 128000  文本开头
            "<|end_of_text|>",                # 128001  文本结尾
            "<|reserved_special_token_0|>",   # 128002  预留
            "<|reserved_special_token_1|>",   # 128003  预留
            "<|reserved_special_token_2|>",   # 128004  预留
            "<|reserved_special_token_3|>",   # 128005  预留
            "<|start_header_id|>",            # 128006  消息头开始
            "<|end_header_id|>",              # 128007  消息头结束
            "<|reserved_special_token_4|>",   # 128008  预留
            "<|eot_id|>",                     # 128009  一轮消息结束 (end of turn)
        ] + [
            # 128010 ~ 128255：246 个纯预留槽位
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        # 建立 {名字: id} 映射：id = 128000 + 在列表中的下标
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        # ③ 创建 tiktoken 编码器（Rust 实现，真正的编解码都由它完成）
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,               # 预切分正则
            mergeable_ranks=mergeable_ranks,    # 普通词表
            special_tokens=self.special_tokens, # 特殊词表
        )

        # ④ 记录常用属性
        self.n_words: int = self.model.n_vocab  # 词表总数 = 128000 + 256 = 128256
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1                   # Llama3 约定 padding 用 -1（不占词表 id）
        self.stop_tokens = {                    # 生成停止符：出现任一个即视为对话结束
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    # ==================== 文本 → token 编号 ====================
    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """把字符串编码成 token 编号列表。

        参数:
            bos/eos: 是否在首尾追加 <|begin_of_text|> / <|end_of_text|>
            allowed_special: 允许哪些特殊 token 字符串被当作"真特殊token"编码
                             （默认空集 = 文本里出现 <|xxx|> 也只当普通文字）
            disallowed_special: 检测到这些特殊 token 字符串就报错（安全防护用）
        """
        assert type(s) is str

        # tiktoken 单次编码超过 40 万字符会触发 Rust panic，
        # 所以官方把超长文本切成小块、逐块编码后拼接。
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000
        # 连续空白或连续非空白超过 2.5 万字符也会出问题（tiktoken issue #195），
        # 因此每块再用下面的静态方法切成「空白段/非空白段」交替的小段。
        MAX_NO_WHITESPACES_CHARS = 25_000

        # 生成器：先按 40 万字符切大块 → 再把每块切成空白/非空白小段
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )

        # 逐块编码，拼成完整 token 列表
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        # 按需在首尾加上 BOS / EOS
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    # ==================== token 编号 → 文本 ====================
    def decode(self, t: Sequence[int]) -> str:
        """把 token 编号列表解码回字符串（特殊 token 会还原成 <|名字|> 字样）。"""
        # tiktoken 内部只按列表取值，这里强转 List[int] 是安全的
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """把字符串切成若干小段，保证每段内「连续空白」或「连续非空白」
        都不超过 max_consecutive_slice_len 个字符。

        例: s='a..bbb   cc', max=2 → ['a.', '.b', 'b ', '  c', 'c']
        """
        current_slice_len = 0                       # 当前段已累计的长度
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0                             # 当前段的起点
        for i in range(len(s)):
            is_now_space = s[i].isspace()
            if current_slice_is_space ^ is_now_space:
                # 空白 <-> 非空白发生切换：重置计数（切换点本身可以留在同一小段里）
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    # 连续同类字符超限 → 在 i 处切断，产出当前段
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]                       # 最后别忘了产出收尾段


# ============================================================
# 第 3 步：ChatFormat 类
# ============================================================
class ChatFormat:
    """把「多轮对话」格式化成 Llama3 要求的 prompt token 序列。

    每条消息的 token 结构（模型训练时就按这个格式，推理必须保持一致）::

        <|start_header_id|>角色<|end_header_id|>\n\n正文<|eot_id|>
        └──────── 消息头 header ────────┘       └─┬─┘ └──┬──┘
                                               正文   消息结束符

    整段对话 = <|begin_of_text|> + 逐条消息 + assistant 消息头
    （最后追加一个空的 assistant 消息头，等模型从这里续写回答）
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        """编码消息头：<|start_header_id|> + 角色 + <|end_header_id|> + \\n\\n"""
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        """编码一条完整消息：消息头 + 正文(去首尾空白) + <|eot_id|>"""
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        """编码整段对话，得到可直接喂给模型的 prompt token 序列。"""
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # 追加 assistant 消息头，模型将从此处开始生成回答
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


# ============================================================
# 演示入口：直接运行本文件即可看到各环节的输出
# ============================================================
if __name__ == "__main__":
    # ① 实例化分词器（首次运行需要几秒钟加载 12.8 万词元）
    tokenizer = Tokenizer(TOKENIZER_PATH)
    chat_format = ChatFormat(tokenizer)
    print(f"词表总数 n_words = {tokenizer.n_words:,}")
    print(f"BOS id = {tokenizer.bos_id}, EOS id = {tokenizer.eos_id}, "
          f"eot_id = {tokenizer.special_tokens['<|eot_id|>']}")

    # ② 直观感受 pat_str 预切分：先把文本切成小片段，再对每个片段做 BPE 合并
    text = "I don't know, but Llama3 counts 42 for 你好世界!"
    print(f"\n预切分演示: {text!r}")
    for piece in regex.findall(Tokenizer.pat_str, text):
        ids = tokenizer.encode(piece, bos=False, eos=False)
        print(f"  {piece!r:20} → {ids}")

    # ③ 纯文本编码 / 解码往返
    ids = tokenizer.encode("Hello, world! 你好，世界！", bos=True, eos=False)
    print(f"\n'Hello, world! 你好，世界！' 编码为 {ids}")
    print(f"解码还原 → {tokenizer.decode(ids)!r}")

    # ④ 特殊 token：默认当普通文字，allowed_special='all' 时才生效
    ids2 = tokenizer.encode("<|begin_of_text|>你好", bos=False, eos=False)
    ids3 = tokenizer.encode("<|begin_of_text|>你好", bos=False, eos=False, allowed_special="all")
    print(f"\n默认模式:    {ids2}   （<|begin_of_text|> 被拆成普通词元）")
    print(f"special模式: {ids3}   （开头 128000 正是真正的 BOS）")

    # ⑤ 长空白分块逻辑验证（一小段人为构造的"超长连续非空白"）
    long_piece = "a" * 30_000  # 3 万个连续非空白字符，超过 2.5 万上限
    chunked = Tokenizer._split_whitespaces_or_nonwhitespaces(long_piece, 25_000)
    print(f"\n分块验证: 30000 个连续字符被切成 {len(list(chunked))} 段（每段 ≤ 25000）")

    # ⑥ 用 ChatFormat 构建多轮对话 prompt
    dialog: Dialog = [
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "用一句话解释什么是分词器。"},
    ]
    prompt_tokens = chat_format.encode_dialog_prompt(dialog)
    print(f"\n对话 prompt 共 {len(prompt_tokens)} 个 token")
    print(f"前 12 个 token id: {prompt_tokens[:12]} ...")

    # ⑦ 反解码，直观看到模型实际"看到"的文本
    print("\n" + "=" * 62)
    print(tokenizer.decode(prompt_tokens))
    print("=" * 62)
