# Unicode 文本与字节序列

> 本章主题：**str 与 bytes——Python 3 最重要的一条类型分界线**
>
> 「人类使用文本，而计算机使用字节。」——Esther Nam 与 Travis Fischer（《Character Encoding and Unicode in Python》，PyCon 2014）
>
> Python 3 明确区分了人类可读的文本字符串（str）与原始字节序列（bytes），彻底禁止了字节与文本之间的隐式转换——这是 Python 3 与 Python 2 最决裂的一处改动，也是无数编码乱码问题的终极答案。本章从「字符到底是什么」出发，覆盖：码点与编码、bytes/bytearray 二进制序列、编解码器、三类编码异常、文件 I/O 的「Unicode 三明治」最佳实践、默认编码陷阱、Unicode 规范化与排序、unicodedata 字符数据库，以及 str/bytes 双模式 API。这一章学完，你不仅能读懂乱码，还能精确解释乱码是怎么产生的。

## 学习地图：本章知识脉络

先建立一张全局心智模型，整章内容都挂在这张图上：

```
                 encode（编码）                  decode（解码）
   str ────────────────────────▶ bytes ────────────────────────▶ str
 （人类文本：码点序列）        （机器字节：0~255 整数序列）      （人类文本：码点序列）
```

- **str 是「概念」，bytes 是「表示」**：同一个字符，在不同编码下字节表示不同（`é` 在 UTF-8 是 2 字节，在 latin1 是 1 字节）。
- **encode 是「打包给机器」，decode 是「翻译给人」**：记忆口诀——把字节序列想象成晦涩的机器核心转储，把 str 想象成人类可读的文本；**解码（decode）得到人类可读的 str，编码（encode）得到用于存储/传输的 bytes**。

全书 12 节可以压缩为 5 条主线，按顺序学即可：

1. **概念层**：字符 → 码点 → 编码 → 字节（4.2–4.4 节）
2. **错误层**：三类编码异常的成因与应对（4.5 节）
3. **实践层**：文本文件 I/O、Unicode 三明治、默认编码陷阱（4.6 节）
4. **文本层**：规范化、大小写折叠、排序——让 Unicode 文本可比较（4.7–4.8 节）
5. **工具层**：unicodedata 数据库与 str/bytes 双模式 API（4.9–4.10 节）

---

## 一、字符问题：从「字符」到「码点」再到「字节」

### 1.1 字符的最佳定义

「字符串就是一串字符」听起来是废话，难在「字符」的定义。如今对字符的最佳定义就是 **Unicode 字符**——Python 3 的 `str` 里取出的每一项都是一个 Unicode 字符（Python 2 的 `unicode` 对象才是它的前身，Python 2 的 `str` 取出的是原始字节）。

Unicode 标准的核心思想：**把字符的标识与字符的字节表示彻底分开**。

- **码点（Code Point）**：字符的标识，是 0 ~ 1,114,111（即 `U+0000` ~ `U+10FFFF`）之间的数字。例如 `A` 是 U+0041，欧元符号 `€` 是 U+20AC，G-clef 音乐符号 𝄞 是 U+1D11E。Unicode 13.0 中只有约 13% 的有效码点真正对应字符。
- **字节表示**：字符实际占用的字节取决于**编码（codec）**——编码是「码点 ↔ 字节序列」之间转换的算法。同一个 `A`（U+0041），UTF-8 编码是 `\x41` 一个字节，UTF-16LE 编码是 `\x41\x00` 两个字节；`€`（U+20AC）在 UTF-8 是 3 字节 `\xe2\x82\xac`，在 UTF-16LE 是 2 字节 `\xac\x20`。

### 1.2 编码与解码：方向千万别记反

```python
>>> s = 'café'
>>> len(s)                  # 4 个 Unicode 字符
4
>>> b = s.encode('utf8')    # str --encode--> bytes
>>> b
b'caf\xc3\xa9'
>>> len(b)                  # 5 个字节：é 的码点 U+00E9 在 UTF-8 中占 2 字节
5
>>> b.decode('utf8')        # bytes --decode--> str
'café'
```

> **记忆要点**
>
> - `encode`：str → bytes，「把人类文本编码成机器字节」；`decode`：bytes → str，「把机器字节解码给人类看」。
> - `len(str)` 数的是**字符数**，`len(bytes)` 数的是**字节数**，二者可以不相等——这是本章一切问题的根源。
> - 一个字符可能占 1~4 个字节，所以**不能用字节偏移去切字符**：`b[3:5].decode('utf8')` 恰好切到 é 的 2 个字节才能解码出 `'é'`，切偏了就是 `UnicodeDecodeError`。

**为什么 Python 3 要如此决绝？（书外背景）** Python 2 中 `str` 与 `unicode` 会隐式互相转换：`'a' + u'b'` 能跑，但一旦混入非 ASCII 字节就在某个遥远的运行时刻炸出 `UnicodeDecodeError`，或默默产出乱码。Python 3 用「类型上直接分开 + 只能显式转换」把问题提前到写代码的瞬间暴露。这是 Python 3 最值得的破坏性改动。

---

## 二、字节要点：bytes 与 bytearray

### 2.1 两种二进制序列类型

Python 内置两种基本的二进制序列类型：

- **`bytes`**：不可变（Python 3 新类型）；
- **`bytearray`**：可变（Python 2.6 引入）。

文档有时用「字节字符串」统称二者，本书避免使用这个混淆术语。最关键的行为差异：

**bytes/bytearray 中的每一项都是 0~255 的整数，而不是单字符字符串；但切片（哪怕长度为 1）仍是同类二进制序列。**

```python
>>> cafe = bytes('café', encoding='utf_8')
>>> cafe
b'caf\xc3\xa9'
>>> cafe[0]           # 索引 → int
99
>>> cafe[:1]          # 切片 → 还是 bytes（哪怕只有一个字节）
b'c'
>>> cafe_arr = bytearray(cafe)   # bytearray 没有字面量语法
>>> cafe_arr
bytearray(b'caf\xc3\xa9')
>>> cafe_arr[-1:]
bytearray(b'\xa9')
```

> **记忆要点**：`my_bytes[0]` 是 `int`，`my_bytes[:1]` 是长度为 1 的 bytes。二者不相等——这和 `str` 的行为（`s[0] == s[:1]`）完全不同，但对 Python 其他序列类型来说，**单项从不等于长度为 1 的切片**（想想 `list`）。不习惯只是因为被 str 惯坏了。

### 2.2 bytes 字面量的四种显示规则

bytes 字面量虽然本质是整数序列，但显示时会尽量「装作」ASCII 文本：

1. 十进制 32~126 的字节（空格到 `~`）→ 直接显示 ASCII 字符；
2. 制表符、换行、回车、反斜杠 → 显示转义序列 `\t`、`\n`、`\r`、`\\`；
3. 若序列中同时出现 `'` 与 `"`，整个序列用 `'` 包裹，内部 `'` 转义为 `\'`；
4. 其余字节 → 十六进制转义（如 `\x00` 空字节）。

所以 `b'caf\xc3\xa9'` 前三个字节落在可打印 ASCII 区间，后两个不在。

### 2.3 支持与不支持的方法

- **不支持**：`format`、`format_map`（格式化），以及依赖 Unicode 数据的方法：`casefold`、`isdecimal`、`isidentifier`、`isnumeric`、`isprintable`、`encode`。
- **支持**：其余 str 方法都在——`endswith`、`replace`、`strip`、`translate`、`upper` 等，只是参数要传 bytes。正则表达式若编译自 bytes（`rb'...'`），也能处理二进制序列。自 Python 3.5 起 `%` 格式化重新支持 bytes（PEP 461）。
- **独有**：类方法 `bytes.fromhex()`，解析空格分隔的十六进制数字对：

```python
>>> bytes.fromhex('31 4B CE A9')
b'1K\xce\xa9'
```

### 2.4 构建二进制序列的方式

除字面量外，`bytes`/`bytearray` 构造函数接受：

1. 一个 str + 关键字参数 `encoding`；
2. 一个可迭代对象，项为 0~255 的整数；
3. 一个实现了缓冲区协议的对象（bytes、bytearray、memoryview、array.array 等）——**复制**源对象的字节序列。

```python
>>> import array
>>> numbers = array.array('h', [-2, -1, 0, 1, 2])   # 'h' = 16 位短整型
>>> octets = bytes(numbers)                          # 复制底层字节
>>> octets
b'\xfe\xff\xff\xff\x00\x00\x01\x00\x02\x00'          # 5 个短整数 = 10 个字节（小端序）
```

用类缓冲对象构建会**复制**数据；与之相对，`memoryview`（见第 2 章 2.10.2）则是在二进制数据结构之间**共享**内存，零拷贝。

> **一句话总结**：bytes 是「不可变的 int 序列，只是显示得像文本」；bytearray 是它的可变版本；memoryview 是不复制、直接借阅底层字节的窗口。

---

## 三、编解码器：文本与字节之间的翻译官

### 3.1 什么是编解码器

Python 发行版自带 **100 多个编解码器（codec）**，每个有标准名（如 `utf_8`）和若干别名（`utf8`、`utf-8`、`U8`），可传给 `open()`、`str.encode()`、`bytes.decode()` 的 `encoding` 参数。同一段文本，不同编码，字节完全不同：

```python
>>> for codec in ['latin_1', 'utf_8', 'utf_16']:
...     print(codec, 'El Niño'.encode(codec), sep='\t')
latin_1  b'El Ni\xf1o'
utf_8    b'El Ni\xc3\xb1o'
utf_16   b'\xff\xfeE\x00l\x00 \x00N\x00i\x00\xf1\x00o\x00'
```

注意 utf_16 的结果开头多了 `b'\xff\xfe'`——这是 BOM（字节序标记），4.5.5 节详解。

### 3.2 常用编码一览

| 编码 | 说明 |
|------|------|
| **latin1**（iso8859_1） | 重要的「基石」编码，cp1252 与 Unicode 的部分映射都以它为基础。字节值 0~255 与码点一一对应 |
| **cp1252** | Microsoft 的 latin1 超集，加了 `{}`、`€` 等符号。部分 Windows 应用称之为「ANSI」，但它并不是 ANSI 标准 |
| **cp437** | IBM PC 最初的字符集（含框线符号），与后来的 latin1 不兼容 |
| **gb2312** | 简体中文的旧标准，亚洲语言多字节编码的代表。书外补充：其扩超集为 GBK，再到 **GB18030**（可表示全部 Unicode 码点，是中国的强制标准） |
| **utf-8** | Web 事实标准。截至 2021 年 7 月，97% 的网站使用 UTF-8（2014 年本书第 1 版出版时为 81.4%） |
| **utf-16le** | 16 位编码方案的小端变体。所有 UTF-16 通过**代理对（Surrogate Pair）**表示 U+FFFF 以上的码点 |

关于 UTF-16 的历史冷知识：1996 年 UTF-16 就取代了最初的 16 位编码 UCS-2。UCS-2 只支持 U+FFFF 以下的码点，但至今仍有系统在用。而截至 2021 年，超过 57% 的已分配码点高于 U+FFFF——包括几乎所有 emoji。**这也是为什么「1 字符 = 2 字节」的 UTF-16 心智模型是错的。**

### 3.3 同一字符在不同编码下的字节（重绘图 4.1）

图 4.1 的核心信息浓缩成一张表（`*` 表示该编码无法表示该字符）：

| 字符 | 码点 | latin1 | cp1252 | gb2312 | utf-8 | utf-16le |
|--------|---------|--------|--------|--------|--------------|--------------|
| A | U+0041 | 41 | 41 | 41 | 41 | 41 00 |
| é | U+00E9 | E9 | E9 | * | C3 A9 | E9 00 |
| ñ | U+00F1 | F1 | F1 | * | C3 B1 | F1 00 |
| € | U+20AC | * | 80 | * | E2 82 AC | AC 20 |
| 中 | U+4E2D | * | * | D6 D0 | E4 B8 AD | 2D 4E |
| 𝄞 | U+1D11E | * | * | * | F0 9D 84 9E | 34 D8 1E DD |

观察规律：**UTF 系列的设计目标就是覆盖全部 Unicode 码点**；ASCII 与 gb2312 这类传统编码只能覆盖子集；UTF-8 对纯 ASCII 文本逐字节兼容。

> **书外补充：UTF-8 的变长编码规则**（理解它，你就能解释为什么 UTF-8 能被「猜」出来、为什么字节切分会炸）：
>
> | 码点范围 | 字节数 | 位模式 |
> |----------|--------|--------|
> | U+0000 ~ U+007F | 1 | `0xxxxxxx` |
> | U+0080 ~ U+07FF | 2 | `110xxxxx 10xxxxxx` |
> | U+0800 ~ U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |
> | U+10000 ~ U+10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |
>
> 验证：`é` = U+00E9 = 二进制 `11101001`，放入 2 字节模板 → `110 00011` `10 101001` = `C3 A9` ✓。关键性质：**多字节序列中绝不会出现 ASCII 字节**（首字节以 `11` 开头，后续字节以 `10` 开头），这让 UTF-8 具有极强抗随机性——随机字节几乎不可能恰好构成合法 UTF-8（4.5.4 节的编码试探技巧就建立在此之上）。

---

## 四、编码与解码问题：三类异常与排查思路

通用异常是 `UnicodeError`，但 Python 报错更具体：**编码（str→bytes）失败抛 `UnicodeEncodeError`，解码（bytes→str）失败抛 `UnicodeDecodeError`**，加载源码文件时还可能是 `SyntaxError`。

> **记忆要点**：收到编码错误，先看异常类型——是 Encode 还是 Decode？这决定了问题出在「你的字符目标编码里没有」还是「你的字节不是你以为的编码」。

### 4.1 处理 UnicodeEncodeError：目标编码装不下这个字符

多数非 UTF 编解码器只覆盖 Unicode 的一小部分。目标编码中没有某个字符时就会报错，除非通过 `errors` 参数指定错误处理器：

```python
>>> city = 'São Paulo'
>>> city.encode('utf_8')
b'S\xc3\xa3o Paulo'
>>> city.encode('iso8859_1')       # latin1 有 ã（U+00E3），没问题
b'S\xe3o Paulo'
>>> city.encode('cp437')           # cp437 没有 ã —— 默认 strict 直接抛异常
Traceback (most recent call last):
UnicodeEncodeError: 'charmap' codec can't encode character '\xe3' in position 1:
character maps to <undefined>
>>> city.encode('cp437', errors='ignore')
b'So Paulo'                        # 静默丢字符——非常糟糕
>>> city.encode('cp437', errors='replace')
b'S?o Paulo'                       # 换成 ?，数据丢但有痕迹
>>> city.encode('cp437', errors='xmlcharrefreplace')
b'S&#227;o Paulo'                  # 换成 XML 实体，数据可恢复
```

错误处理器速查：

| 处理器 | encode 时行为 | 评价 |
|--------|---------------|------|
| `strict`（默认） | 抛异常 | **首选**。尽早暴露问题 |
| `ignore` | 跳过无法编码的字符 | 数据无声丢失，通常是最坏选择 |
| `replace` | 用 `?` 代替 | 有提示但仍丢数据 |
| `xmlcharrefreplace` | 替换为 `&#227;` 式 XML 实体 | 不能用 UTF 又不能丢数据时的唯一正解 |
| `backslashreplace`（书外补充） | 替换为 `\xe3` 式转义序列 | 日志、调试场景极好用 |
| `namereplace`（书外补充，3.5+） | 替换为 `\N{LATIN SMALL LETTER A WITH TILDE}` | 调试时比十六进制更可读 |

错误处理是可扩展的：`codecs.register_error(name, handler)` 可注册自定义处理器。

另一个实用结论：**ASCII 是所有主流编码的公共子集**——纯 ASCII 文本用任何编码 encode 都不会炸。Python 3.7 新增 `str.isascii()`，为 True 时可放心用任意编码。

### 4.2 处理 UnicodeDecodeError：字节不是你以为的编码

并非每个字节序列都是合法的 UTF-8/UTF-16，假定错了就抛 `UnicodeDecodeError`。更阴险的是：**许多传统 8 位编码（cp1252、iso8859_1、koi8_r）什么字节流都能解、从不报错**——用错编码时它们会默默产出乱码。乱码字符被称为「鬼符（gremlin）」，日语称 **mojibake（文字化け）**。

```python
>>> octets = b'Montr\xe9al'          # 用 latin1 编码的 'Montréal'，\xe9 = é
>>> octets.decode('cp1252')          # cp1252 是 latin1 超集，正确
'Montréal'
>>> octets.decode('iso8859_7')       # 希腊文编码：不报错，但解出希腊字母
'Montrιal'
>>> octets.decode('koi8_r')          # 俄文编码：不报错，解出西里尔字母
'MontrИal'
>>> octets.decode('utf_8')           # \xe9 不是合法 UTF-8 续字节 → 报错
Traceback (most recent call last):
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 5:
invalid continuation byte
>>> octets.decode('utf_8', errors='replace')
'Montr\ufffdal'                      # U+FFFD 是官方「替换字符」�
```

注意这个反直觉现象：**报 UTF-8 的错反而是好事**——说明字节真的不是 UTF-8；而解码「成功」的 cp1252/iso8859_7/koi8_r 里只有一个是真对的，其余全是乱码。**异常是你的朋友，静默乱码才是敌人。**

### 4.3 加载模块时的 SyntaxError

Python 3 源码默认 UTF-8。若 `.py` 文件是其他编码（比如在 Windows 上用 cp1252 保存）且未声明编码，加载时报：

```
SyntaxError: Non-UTF-8 code starting with '\xe1' in file ola.py on line 1,
but no encoding declared; see https://python.org/dev/peps/pep-0263/ for details
```

解法是在文件顶部加编码声明注释（PEP 263）：

```python
# coding: cp1252
print('Olá, Mundo!')
```

但更好的「修复」是把旧源码转成 UTF-8，从此不再需要编码注释。编辑器不支持 UTF-8？换编辑器。

### 4.4 如何找出字节序列的编码

**理论上：不可能。** 编码信息必须由外部（协议头、文件元数据、发送方）告诉你。但实践中有线索可用：

- HTTP、XML 等协议/格式可在头部声明编码；
- 含大于 127 的字节 → 肯定不是 ASCII；
- `\x00` 频繁出现 → 大概率是 16/32 位编码（纯文本不会有空字节）；
- `\x20\x00` 频繁出现 → 大概率是 UTF-16LE 的空格（U+0020）；
- **UTF-8 试探法**（技术审校 Leonardo Rochael 的实战经验）：因为 UTF-8 多字节序列的位模式极其严格（见 3.3 节补充），随机字节或非 UTF-8 文本几乎不可能蒙混过关。所以「先用 UTF-8 试解，报 UnicodeDecodeError 再退回 cp1252」这个不优雅的策略在实践中非常有效；
- 工具化方案：**Chardet** 库（可识别 30 多种编码），附带命令行工具：

```
$ chardetect 04-text-byte.asciidoc
04-text-byte.asciidoc: utf-8 with confidence 0.99
```

> 书外补充：requests 库自 2.26 起已用 `charset-normalizer` 替代 chardet 作为默认编码探测后端，API 兼容、速度更快。

### 4.5 BOM：字节序标记

UTF-16 编码 `utf_16` 时输出开头总带着 `b'\xff\xfe'`，这就是 **BOM（Byte-Order Mark，字节序标记）**，用于声明编码时使用的字节序：

- 小端（little-endian，Intel CPU）：码点最低有效字节在前。字母 `E`（U+0045 = 十进制 69）编码为 `69, 0`；
- 大端（big-endian）：反过来，`0, 69`。

为避免混淆，UTF-16 编码器会在文本前插入不可见字符 **U+FEFF（ZERO WIDTH NO-BREAK SPACE）**：小端下它是 `b'\xff\xfe'`。由于 Unicode 标准故意不定义 U+FFFE，所以 `b'\xff\xfe'` 开头必然意味着 U+FEFF + 小端——解码器据此识别字节序。

两个要点：

1. `utf_16le` / `utf_16be` 显式指定字节序，**不生成 BOM**。按 Unicode 标准，无 BOM 的 UTF-16 文件应假定为大端——但现实中 x86 世界充满了无 BOM 的小端文件；
2. **字节序只影响多字节编码（UTF-16/UTF-32）**；UTF-8 无论设备字节序如何，输出永远一致，本不需要 BOM。但某些 Windows 程序（记事本、Excel）会给 UTF-8 文件强加 BOM——这种变体在 Python 编解码器注册表中叫 **`utf-8-sig`**：U+FEFF 被编码为 3 字节 `b'\xef\xbb\xbf'`，文件以此开头大概率是带 BOM 的 UTF-8。

> **技术审校 Caleb Hattingh 的建议**：读取 UTF-8 文件时**始终用 `utf-8-sig`**——无论文件有没有 BOM 都能正确读取且不返回 BOM 本身，无害。但写入时建议用纯 `utf-8`：BOM 会破坏 `#!/usr/bin/env python3` shebang 约定（文件前两字节必须是 `#!`），Python 官方文档也「不鼓励在 UTF-8 中使用 BOM」。需要给 Excel 等依赖 BOM 探测的软件导出数据时，才用 utf-8-sig。
>
> 书外补充：UTF-32 的 BOM 是 4 字节（小端 `FF FE 00 00`），同样利用了 U+FFFE 未定义的性质。

---

## 五、处理文本文件：Unicode 三明治与默认编码陷阱

### 5.1 最佳实践：Unicode 三明治

处理文本 I/O 的黄金法则叫 **Unicode 三明治**（Ned Batchelder，PyCon 2012《Pragmatic Unicode》）：

```
 bytes ──decode──▶ str ══▶ 业务逻辑（全部用 str）══▶ str ──encode──▶ bytes
 输入边界：尽早解码                                  输出边界：尽晚编码
```

- 应用核心（三明治的「馅」）只处理 str，绝不混入编码/解码操作；
- 只在输入边界尽早 decode，输出边界尽晚 encode；
- Web 框架都是这么干的：Django 视图输出 str，框架负责以 UTF-8 编码为 bytes。

Python 3 让三明治天然容易：`open()` 文本模式读取时自动解码、写入时自动编码，`my_file.read()` 给你 str，`my_file.write(text)` 收 str。

### 5.2 经典 bug 剖析：写读编码不一致

```python
# 在 64 位 Windows 10 + Python 3.8 上运行：
>>> open('cafe.txt', 'w', encoding='utf_8').write('café')
4
>>> open('cafe.txt').read()
'cafÃ©'          # ？？乱码
```

bug：**写入时显式 UTF-8，读取时省略 encoding**，Python 回退到 Windows 默认的 cp1252：UTF-8 把 `é` 编码为 `0xC3 0xA9`，cp1252 把 `0xC3` 解读为 `Ã`、`0xA9` 解读为 `©`——乱码 `cafÃ©` 诞生。而在 GNU/Linux / macOS 上同样代码「正常」，因为默认编码恰好也是 UTF-8——**这不是正确，只是侥幸**。

示例 4.9 的完整解剖还揭示了一个细节：`write` 返回 4（字符数），`os.stat(...).st_size` 是 5（字节数）——`len(str)` 与字节数从此要分家：

```python
>>> fp = open('cafe.txt', 'w', encoding='utf_8')
>>> fp                                  # 文本模式返回 TextIOWrapper
<_io.TextIOWrapper name='cafe.txt' mode='w' encoding='utf_8'>
>>> fp.write('café'); fp.close()
4
>>> os.stat('cafe.txt').st_size         # 磁盘上是 5 个字节
5
>>> open('cafe.txt').encoding           # 不指定 encoding 时的默认值
'cp1252'                                # （Windows 上）
>>> open('cafe.txt', encoding='utf_8').read()
'café'                                  # 指对编码，一切正常
>>> open('cafe.txt', 'rb').read()       # 二进制模式返回 BufferedReader，读出原始字节
b'caf\xc3\xa9'
```

> **记忆要点**：除非检查文件编码，否则**不要用二进制模式打开文本文件**（想猜编码请用 Chardet，别造轮子）；`'rb'` 的正当用途是处理二进制文件（图片、协议包等）。

### 5.3 默认编码从哪里来

影响 Python I/O 默认编码的设置有好几个（书中用脚本 `default_encodings.py` 逐一探测），核心结论如下表：

| 设置 | 作用 | GNU/Linux / macOS | Windows（典型） |
|------|------|--------------------|------------------|
| `locale.getpreferredencoding()` | **最重要**：`open()` 文本模式的默认编码；stdout/stderr 重定向到文件时的编码 | UTF-8 | cp1252 |
| `sys.stdout/stdin/stderr.encoding` | 标准流编码 | utf-8 | 控制台下 utf-8（PEP 528）；重定向到文件时由 locale 决定 |
| `sys.getdefaultencoding()` | Python 内部 str↔bytes 隐式转换用（几乎不接触，**别去改**） | utf-8 | utf-8 |
| `sys.getfilesystemencoding()` | 文件名（不是文件内容！）的编解码 | utf-8 | utf-8（PEP 529 起，此前是 MBCS） |

几个关键事实：

- Windows 上 `chcp` 显示的控制台代码页（如 437）与 `sys.stdout.encoding`（utf-8）可以不同——PEP 528 让 Windows 控制台的标准流改用 UTF-8，于是现在 Windows 上直接 `print` Unicode 一般不炸；**但输出重定向到文件时**，编码又变回 `locale.getpreferredencoding()`，此时 print 不在目标代码页里的字符（如 `∞` 不在 cp1252 中）就会抛 `UnicodeEncodeError`。同一台 Windows 机器上多套互不兼容的编码并存，正是「Windows 糟糕 Unicode 体验」的根源；
- `PYTHONIOENCODING` 环境变量在 Python 3.6+ 的 Windows 控制台场景被忽略（除非设 `PYTHONLEGACYWINDOWSSTDIO`）；
- `locale.getpreferredencoding()` 官方文档自己都说它只是「返回一个猜测的编码」。

### 5.4 黄金法则

> **需要在多台设备或多种场合运行的代码，绝不能依赖 encoding 默认值。打开文件时永远显式写 `encoding='utf-8'`。**

```python
# 永远这样写：
with open('data.txt', 'w', encoding='utf-8') as fp:
    fp.write(text)

with open('data.txt', encoding='utf-8') as fp:      # 读带 BOM 的文件可改用 'utf-8-sig'
    content = fp.read()
```

> **书外补充（现代实践）**：
>
> - **UTF-8 模式（PEP 540，Python 3.7+）**：设环境变量 `PYTHONUTF8=1` 或用 `python -X utf8` 启动，可把 `open()` 等默认编码全局强制为 UTF-8，是过渡期的实用逃生舱；
> - **PEP 686** 已决定在未来的 Python 3.15 中将 UTF-8 模式设为默认——届时「默认编码陷阱」将成为历史，但在那之前的所有 Python 版本上，显式指定 `encoding` 仍是唯一可靠做法；
> - Windows 10/11 也可在系统区域设置中勾选「Beta: 使用 Unicode UTF-8 提供全球语言支持」，从操作系统层面缓解。

---

## 六、Unicode 规范化：让「看起来相同」的字符串真正相等

### 6.1 问题：同一字符有多种码点序列

Unicode 存在**组合字符**（附加在前一字符上的变音符等标记）。于是 `café` 有两种合法写法：4 个码点的预组合形式，或 5 个码点的分解形式（`e` + U+0301 COMBINING ACUTE ACCENT）。显示上一模一样，Python 却认为它们不同：

```python
>>> s1 = 'café'                        # 预组合 é（U+00E9）
>>> s2 = 'cafe\N{COMBINING ACUTE ACCENT}'   # e + 组合尖音符
>>> s1, s2
('café', 'café')                       # 看起来一样
>>> len(s1), len(s2)
(4, 5)
>>> s1 == s2
False                                  # 码点序列不同 → 不相等
```

Unicode 标准把这样的序列称为**规范等价物（Canonical Equivalents）**，应用程序应视其为相同；但 Python 看到的是不同码点序列。解药：`unicodedata.normalize()`，第一个参数是 `NFC` / `NFD` / `NFKC` / `NFKD` 四选一。

### 6.2 四种规范化形式

| 形式 | 全称 | 做什么 | 用途 |
|------|------|--------|------|
| **NFC** | Normalization Form C | **组合**码点，生成最短等价字符串 | **推荐默认**；W3C 推荐；键盘输入通常已是 NFC |
| **NFD** | Normalization Form D | **分解**合成字符为基字符 + 组合字符 | 与 NFC 等价可互比；去变音符的预处理 |
| **NFKC** | Normalization Form **K**ompatibility C | 兼容性**分解**再组合 | 搜索/索引的中间表示 |
| **NFKD** | Normalization Form **K**ompatibility D | 兼容性分解 | 同上 |

NFC 与 NFD 都能实现正确比较：

```python
>>> from unicodedata import normalize
>>> normalize('NFC', s1) == normalize('NFC', s2)
True
>>> normalize('NFD', s1) == normalize('NFD', s2)
True
```

NFC 还会把一些单字符改写为等价单字符：电阻单位 `Ω`（U+2126，OHM SIGN）会被规范化为大写希腊字母 `Ω`（U+03A9，GREEK CAPITAL LETTER OMEGA）——视觉相同但码点不同、比较为 False，所以**保存用户输入前先 `normalize('NFC', text)` 是安全习惯**。

NFKC/NFKD 更激进，会处理「兼容性字符」——为兼容旧标准而重复收录的字符：

```python
>>> from unicodedata import normalize, name
>>> half = '\N{VULGAR FRACTION ONE HALF}'    # ½
>>> normalize('NFKC', half)
'1⁄2'          # 注意：中间是 U+2044 FRACTION SLASH，不是 ASCII 的 '/'！
>>> micro = '\N{MICRO SIGN}'                  # µ 微符号（兼容性字符，U+00B5）
>>> name(normalize('NFKC', micro))
'GREEK SMALL LETTER MU'                       # 变成希腊字母 μ（U+03BC）
>>> normalize('NFKC', '4²')                   # 上标 2
'42'
```

> **记忆要点**：NFKC/NFKD 兼容性分解会**改变甚至曲解原意**（`4²` 变 `42`），还会引入 `1⁄2` 这种搜索 `'1/2'` 找不到的坑。**NFKC/NFKD 只用于搜索、索引等场景，绝不能用于永久存储**。日常比较用 NFC 即可。

### 6.3 大小写同一化：casefold

`str.casefold()` 是「更彻底的 lower」：把文本统一为小写并附加一些转换。对 latin1 文本，它与 `s.lower()` 几乎一样，仅两处例外：

```python
>>> micro = 'µ'
>>> micro.casefold()          # MICRO SIGN → GREEK SMALL LETTER MU
'µ'
>>> 'ß'.casefold()            # 德语 Eszett：ß → ss（长度都变了！）
'ss'
```

casefold 与 lower 结果不同的码点有近 300 个。**做不区分大小写的比较时，用 casefold 而不是 lower。**

### 6.4 实用函数：规范化比较的工具箱

书中给出的两个工具函数值得直接抄进工具库（示例 4.13）：

```python
from unicodedata import normalize

def nfc_equal(str1, str2):
    """NFC 规范化后比较（区分大小写）"""
    return normalize('NFC', str1) == normalize('NFC', str2)

def fold_equal(str1, str2):
    """NFC 规范化 + casefold 后比较（不区分大小写）"""
    return (normalize('NFC', str1).casefold()
            == normalize('NFC', str2).casefold())
```

效果：`nfc_equal('café', 'cafe\u0301')` → True；`fold_equal('Straße', 'strasse')` → True；`fold_equal('A', 'a')` → True。

### 6.5 极端规范化：去掉变音符（shave_marks / asciize）

Google 搜索的秘诀之一就是忽略变音符（搜 `cafe` 能命中 `café`）。去变音符虽不「正确」（可能改变词义、产生误报），但对搜索友好，也能让 URL 更可读——维基百科 `São Paulo` 的 URL 是 `%C3%A3o`（URL 编码的 UTF-8），而 `Sao_Paulo` 拼写虽不对却更好认。

核心三板斧（示例 4.14）：**NFD 分解 → 过滤组合字符 → NFC 重组**：

```python
import unicodedata

def shave_marks(txt):
    """去除所有变音符"""
    norm_txt = unicodedata.normalize('NFD', txt)                 # ① 全部分解
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))            # ② 过滤组合标记
    return unicodedata.normalize('NFC', shaved)                  # ③ 重新组合
```

但它会「用力过猛」——希腊字母 `Ζέφυρος` 也被剥成 `Ζεφυρος`，而希腊字符去掉变音符永远变不成 ASCII。改进版 `shave_marks_latin` 只对拉丁基字符移除组合标记。更激进的 `asciize`（示例 4.17）还会先用 `str.translate` 把 cp1252 特有符号（花引号 `“”`、长破折号 `—`、`™`、`€` 等）替换为 ASCII 对应物，再把 `ß` 换成 `ss`，最后 NFKC 收尾：

```python
def asciize(txt):
    no_marks = shave_marks_latin(dewinize(txt))   # 先换符号，再去变音符
    no_marks = no_marks.replace('ß', 'ss')        # 保留大小写，故不用 casefold
    return unicodedata.normalize('NFKC', no_marks)
```

其中 `dewinize` 用 `str.maketrans` 的两种形式建替换表——**「字符到字符」**（`str.maketrans('abc', 'xyz')`）与**「字符到字符串」**（`str.maketrans({'€': 'EUR', '…': '...'})`），后者必须传字典。`translate` 是高效的批量字符替换工具。

> **记忆要点**：这类深度改写会改变文本原意，**只有了解目标语言、目标用户和用途时才做**（如德语 `ü` 的 ASCII 化规则是 `ue` 而非 `u`，通用函数必然顾此失彼）。

---

## 七、Unicode 文本排序

### 7.1 码点排序的问题

Python 排序字符串就是逐个比较码点。对非 ASCII 文本，结果不可接受：

```python
>>> fruits = ['caju', 'atemoia', 'cajá', 'açaí', 'acerola']
>>> sorted(fruits)
['acerola', 'atemoia', 'açaí', 'caju', 'cajá']    # 码点序：带重音的 á 排在 u 之后
```

葡萄牙语（以及多数拉丁语言）习惯把变音符当透明处理：`cajá` 按 `caja` 排序，应排在 `caju` 之前。正确结果应为 `['açaí', 'acerola', 'atemoia', 'cajá', 'caju']`。

### 7.2 方案一：locale.strxfrm

标准库做法是用 `locale.strxfrm` 作为排序键——它「把字符串转换为可用于本地化比较的字符串」：

```python
import locale
locale.setlocale(locale.LC_COLLATE, 'pt_BR.UTF-8')      # 必须先设 locale
sorted(fruits, key=locale.strxfrm)
# ['açaí', 'acerola', 'atemoia', 'cajá', 'caju']  ✓
```

但有四个坑：

- `locale` 设置**全局生效**，不要在库中调用 `setlocale`；应用/框架应在进程启动时设定且不再更改；
- 系统必须安装对应 locale，否则抛 `locale.Error`；
- locale 名称拼写因系统而异；
- **操作系统实现质量参差**：作者在 Ubuntu 上成功，在 macOS 10.14 上 `strxfrm` 完全不生效（部分人报告可用）。

结论：标准库方案只在 GNU/Linux 上可靠，还把部署绑定到系统 locale——所以有更简单的选择。

### 7.3 方案二：pyuca（纯 Python 的 Unicode 排序算法）

```python
>>> import pyuca
>>> coll = pyuca.Collator()
>>> sorted(fruits, key=coll.sort_key)
['açaí', 'acerola', 'atemoia', 'cajá', 'caju']    # 跨平台正确
```

pyuca 纯 Python 实现了 **UCA（Unicode Collation Algorithm，Unicode 排序算法）**，不依赖系统 locale，GNU/Linux/macOS/Windows 通吃。默认使用 Unicode 官方排序表 `allkeys.txt`（Default Unicode Collation Element Table 的副本），也可传入自定义排序表。

> 技术审校 Miroslav Šedivý 的补充：pyuca 不考虑具体语言规则（德语 `Ä` 排在 A、B 之间，瑞典语则排在 Z 之后；土耳其语的 i/İ 大小写转换也是特例）。需要语言级精确排序时用 **PyICU**——像 locale 一样工作但无需改进程 locale，代价是含 C++ 扩展、安装较难。

---

## 八、Unicode 数据库：unicodedata 模块

Unicode 标准自带一个完整数据库（一组结构化文本文件），不只存「码点 → 字符名」映射，还存每个字符的元数据与关系——是否可打印、是否字母、是否十进制数字等。`str.isprintable`、`isalpha`、`isdecimal`、`isnumeric` 和 `casefold` 的行为全部来自这套数据。

### 8.1 按名称查找字符

`unicodedata.name(char)` 返回字符的官方名称，`unicodedata.category(char)` 返回两位字母的类别码。据此可以写一个 28 行的字符搜索工具 `cf.py`（示例 4.21）——按名称关键词搜索全部码点：

```python
#!/usr/bin/env python3
import sys
import unicodedata

START, END = ord(' '), sys.maxunicode + 1        # 搜索范围的默认值

def find(*query_words, start=START, end=END):
    query = {w.upper() for w in query_words}
    for code in range(start, end):
        char = chr(code)
        name = unicodedata.name(char, None)      # 未分配的码点返回 None
        if name and query.issubset(name.split()):
            print(f'U+{code:04X}\t{char}\t{name}')

def main(words):
    if words:
        find(*words)
    else:
        print('Please provide words to find.')

if __name__ == '__main__':
    main(sys.argv[1:])
```

亮点是用**集合的 `issubset()`** 一行完成「所有查询词都出现在字符名中」的判断，免掉嵌套 for 循环——再次体会第 3 章集合 API 的表达力。另外注意 `\N{官方名称}` 转义写法：字符名写错会直接抛 `SyntaxError`，比写错十六进制数（静默出错）友好得多。

### 8.2 数字的多种含义：isdecimal / isdigit / isnumeric / numeric()

Unicode 里有大量「表示数字」的字符，Python 提供了三个由宽到窄的判断方法，加一个取值的函数：

| 字符 | isdecimal | isdigit | isnumeric | unicodedata.numeric() |
|------|:---:|:---:|:---:|:---:|
| `1`（U+0031 阿拉伯数字） | ✓ | ✓ | ✓ | 1.0 |
| `１`（U+FF11 全角数字） | ✓ | ✓ | ✓ | 1.0 |
| `३`（U+0969 梵文数字 3） | ✓ | ✓ | ✓ | 3.0 |
| `²`（U+00B2 上标 2） | ✗ | ✓ | ✓ | 2.0 |
| `½`（U+00BD 分数） | ✗ | ✗ | ✓ | 0.5 |
| `Ⅻ`（U+216B 罗马数字 12） | ✗ | ✗ | ✓ | 12.0 |
| `四`（汉字数字） | ✗ | ✗ | ✓ | 4.0 |
| `a` | ✗ | ✗ | ✗ | ValueError |

三者是严格的包含关系：**isdecimal ⊂ isdigit ⊂ isnumeric**。`unicodedata.numeric()` 甚至知道罗马数字 `Ⅻ` 等于 12.0——想做一个支持泰米尔数字或罗马数字的电子表格？素材齐了。

另一个重要发现：正则 `r'\d'` 能匹配阿拉伯数字与梵文数字（两者都是 Nd 十进制类别），却匹配不上 `isdigit` 认可的 `²`、`½`——**re 模块对 Unicode 的支持并不完整**（PyPI 的 `regex` 模块旨在替代 re，Unicode 支持更好）。

---

## 九、双模式 API：str 与 bytes 通吃

标准库中有些函数同时接受 str 或 bytes 参数，且**行为随类型而变**。代表：re 与 os 模块。

### 9.1 正则表达式中的 str 与 bytes

用 str 编译的正则，`\d`、`\w` 能匹配 ASCII 之外的 Unicode 数字/字母；用 bytes 编译则**只匹配 ASCII**：

```python
import re
re_numbers_str = re.compile(r'\d+')        # str 模式
re_numbers_bytes = re.compile(rb'\d+')     # bytes 模式

text_str = 'Ramanujan saw ௧௭௨௯ as 1729 = 1³ + 12³ = 9³ + 10³.'
text_bytes = text_str.encode('utf_8')

re_numbers_str.findall(text_str)
# ['௧௭௨௯', '1729', '1', '12', '9', '10']    ← 泰米尔数字也能匹配
re_numbers_bytes.findall(text_bytes)
# ['1729', '1', '12', '9', '10']            ← 只有 ASCII 数字
```

规则总结：**bytes 正则眼里，ASCII 之外的字节都是非数字、非单词字符**。若想反过来让 str 正则只匹配 ASCII，用 `re.ASCII` 标志。

### 9.2 os 函数中的 str 与 bytes

GNU/Linux 内核对 Unicode 支持不完善，现实中存在「任何编码都解不出」的文件名（连多种操作系统互连的文件服务器上尤其常见）。为此，**所有接受文件名/路径的 os 函数都是双模式的**：

- 传 `str`：按 `sys.getfilesystemencoding()` 自动编解码——符合 Unicode 三明治，首选；
- 传 `bytes`：原样进出，返回 bytes——让你能处理任何「鬼符」文件名：

```python
>>> os.listdir('.')
['abc.txt', 'digits-of-π.txt']
>>> os.listdir(b'.')
[b'abc.txt', b'digits-of-\xcf\x80.txt']    # π 的 UTF-8 编码 = \xcf\x80
```

配套工具：`os.fsencode(name_or_path)` / `os.fsdecode(name_or_path)`（3.6 起接受 str、bytes 或任何 `os.PathLike` 对象），用于手动在两种表示间转换。

> 书外补充：文件系统编解码器实际搭配的错误处理器是 `surrogateescape`——解不出的字节被映射到 U+DC80~U+DCFF 代理区暂存，重新编码时原样还原。这就是「str 文件名也从不丢字节」的底层机制，也是 `errors='surrogateescape'` 处理器的出处。

---

## 十、杂谈三则（书中的延伸彩蛋）

### 10.1 str 的码点在内存中如何表示（PEP 393）

自 Python 3.3 起，创建 str 对象时解释器会扫描内容，选择最经济的内部布局：纯 latin1 范围 → 每字符 1 字节；否则 2 或 4 字节。类似 int 的自适应表示——机器字放得下就紧凑存，放不下换变长表示。

副作用（Armin Ronacher 指出）：往纯 ASCII 文本里插入一个 emoji（如 RAT U+1F400），整个字符串内部立刻膨胀到每字符 4 字节；且由于组合字符的存在，「按位置快速取字符/切片」在极端情形下并不可靠。**理论上内部表示不重要——反正输出时总要编码成字节**；但知道这一点，你就理解了为什么 `sys.getsizeof('a')` 和 `sys.getsizeof('a' + '\U0001F400')` 差异巨大。

### 10.2 什么是「纯文本」

Unicode 术语表：纯文本是「仅由给定标准的一系列码点组成、没有其他格式或结构信息的计算机编码文本」。作者的理解更实用：HTML 也是纯文本——它每个字节都表示文本字符（通常 UTF-8）；而 `.png`/`.xls` 中大多数字节是打包的二进制值（RGB、浮点数）。纯文本中数字也表示为数字字符序列。

### 10.3 源代码中的非 ASCII 标识符

Python 3 允许 `ação = 'PBR'`、`ε = 10**-6`。作者（巴西人）的观点：**选择让目标读者最容易阅读的语言命名，然后用正确的字符拼写**——面向国际开源项目就用英文 ASCII；面向本国学生，省略重音的 `acao` 并不比 `ação` 更可读。

---

## 本章总结

### 核心概念速查表

| 概念 | 一句话定义 |
|------|-----------|
| 码点 Code Point | 字符的标识数字，U+0000 ~ U+10FFFF |
| 编码 encode | 码点序列 → 字节序列（str → bytes） |
| 解码 decode | 字节序列 → 码点序列（bytes → str） |
| bytes / bytearray | 不可变/可变的 0~255 整数序列；索引得 int，切片得同类 |
| 编解码器 codec | 「码点 ↔ 字节」的转换算法，100+ 内置，如 utf_8 |
| BOM | U+FEFF 的编码（UTF-16LE 为 `\xff\xfe`；UTF-8-SIG 为 `\xef\xbb\xbf`），标记字节序 |
| Unicode 三明治 | 尽早 decode → 核心逻辑只用 str → 尽晚 encode |
| NFC / NFD | 组合 / 分解的规范化形式，用于安全比较 |
| NFKC / NFKD | 兼容性分解，仅用于搜索/索引，会丢信息 |
| casefold | 比 lower 更彻底的大小写折叠（ß→ss），用于大小写不敏感比较 |
| UCA / pyuca | Unicode 排序算法及其纯 Python 实现，跨平台正确排序 |
| unicodedata | Unicode 数据库接口：name()、category()、combining()、numeric() 等 |

### 最佳实践清单

1. **打开文本文件永远显式传 `encoding='utf-8'`**（读可能带 BOM 的文件用 `utf-8-sig`）——不依赖默认编码；
2. 遵循 **Unicode 三明治**：程序核心只有 str，边界处显式编解码；
3. 不要用二进制模式打开文本文件（`'rb'` 留给真正的二进制文件）；
4. 保存/比较用户输入前先 `unicodedata.normalize('NFC', s)`；大小写不敏感比较用 `casefold()`；
5. 非 ASCII 排序用 `pyuca`（跨平台）而非 `locale.strxfrm`（依赖系统 locale）；
6. 处理编码错误首选 `strict`（让异常尽早暴露）；要容错时 encode 用 `xmlcharrefreplace`/`backslashreplace`，decode 用 `replace`；永远警惕 `ignore`；
7. 字节序列不确定编码时：先 UTF-8 试解（失败再退 cp1252/latin1），或直接上 Chardet/charset-normalizer；
8. NFKC/NFKD 只用于搜索索引，绝不用于存储。

### 常见坑 Top 5

1. **`b[0]` 是 int 不是 bytes**——二进制序列的单项 ≠ 长度 1 的切片；
2. **Windows 上读写编码不一致**→ `cafÃ©` 式乱码（写 UTF-8 读 cp1252）；Linux/macOS 只是碰巧默认 UTF-8，代码照样不正确；
3. **传统 8 位编码解码从不报错**——用错编码悄悄产出 mojibake；报 `UnicodeDecodeError` 反而是及时止损；
4. **`é` 的两种写法不相等**——不做 NFC 规范化，字符串比较、字典键、集合去重都会出错；
5. **`len(str)` 是字符数，`len(bytes)` 是字节数**——按字节偏移切 UTF-8 字符串必然切出半个字符。

---

## 自测：检验你是否真的懂了

尝试不回看上文回答，再看答案：

1. `'café'.encode('utf8')` 的结果是什么？`len` 为什么是 5？
2. `b'café'` 语法合法吗？`b'caf\xc3\xa9'[0]` 的类型和值？
3. `encode` 和 `decode` 各自的方向？哪个把字节变人类可读？
4. `open('f.txt', 'w').write('中文')` 在 Windows 上可能产生什么问题？如何杜绝？
5. 为什么 `b'Montr\xe9al'.decode('koi8_r')` 不报错却输出乱码，而 `decode('utf_8')` 报错？
6. BOM 是哪个字符？`utf_16le` 编码会产生 BOM 吗？读带 BOM 的 UTF-8 文件该用什么编码名？
7. `'café'` 和 `'cafe\u0301'` 为什么不相等？如何让它们相等？再让 `Straße == 'strasse'` 成立呢？
8. NFKC 把 `½` 变成什么？为什么 NFKC 不能用于永久存储？
9. bytes 模式的正则 `rb'\d+'` 能匹配泰米尔数字 `௧` 吗？为什么？
10. `os.listdir(b'.')` 传 bytes 参数的意义是什么？

**参考答案**

1. `b'caf\xc3\xa9'`。`é`（U+00E9）的 UTF-8 编码是 2 字节（C3 A9），4 个字符变成 5 个字节。
2. 不合法——bytes 字面量只能含 ASCII 字符。`[0]` 是 `int`，值 99（`c` 的 ASCII 码）；`[:1]` 才是 `b'c'`。
3. `encode`: str→bytes；`decode`: bytes→str。**decode** 把字节解码为人类可读文本。
4. 未指定 encoding 时用 `locale.getpreferredencoding()`（Windows 常为 cp1252），文件实际按 cp1252 写出，跨平台/跨机器不兼容。杜绝方法：永远显式 `encoding='utf-8'`（或启用 PYTHONUTF8=1）。
5. koi8_r 这类 8 位编码定义了全部 256 个字节值的映射，什么都能解（解出来是错的）；UTF-8 有严格的位模式校验，`\xe9` 不是合法续字节所以报错。报错是好事。
6. U+FEFF（ZERO WIDTH NO-BREAK SPACE）。不会——`utf_16le`/`utf_16be` 显式指定字节序、不产生 BOM；只有默认的 `utf_16` 会加。读 UTF-8 文件用 `utf-8-sig`（有无 BOM 均可正确读取且不返回 BOM）。
7. 两者码点序列不同（4 个 vs 5 个码点），Python 按码点比较。用 `unicodedata.normalize('NFC', s)`（或 NFD）后再比较；不区分大小写再加 `.casefold()`（NFC + casefold，即书中 `fold_equal`）。
8. 变成 `1⁄2`（注意中间是 U+2044 FRACTION SLASH 而非 `/`）。因为兼容性分解会改变原意（`4²`→`42`、µ→μ），且引入搜索不一致，只适合搜索/索引的临时中间表示。
9. 不能。bytes 正则中 `\d` 只匹配 ASCII 数字字节；要匹配 Unicode 数字需用 str 模式（或 `regex` 模块）。
10. 告诉 os 函数「别帮我解码」：以 bytes 原样返回文件名，从而能处理任何编码都解不出的「鬼符」文件名（配合 `os.fsdecode` 手动处理）。

---

## 延伸阅读精选（书中 4.12 节浓缩）

- **Ned Batchelder《Pragmatic Unicode, or, How Do I Stop the Pain?》**（PyCon 2012）——「Unicode 三明治」出处，本章最佳入门演讲；
- **Python 官方文档《Unicode HOWTO》**——历史、语法、编解码器、文件 I/O 最佳实践全覆盖；
- **《Dive Into Python 3》第 4 章「字符串」**——Python 3 Unicode 支持的经典介绍；第 15 章的 Chardet 移植案例展示了 str→bytes 迁移的全部痛点；
- **Nick Coghlan《Python Notes》** 两篇：《Python 3 and ASCII Compatible Binary Protocols》《Processing Text Files in Python 3》，强烈推荐；
- 书外补充：**PEP 393**（灵活字符串表示）、**PEP 540**（UTF-8 模式）、**PEP 686**（UTF-8 默认化）——把本章「杂谈」与现代演进串起来读。
