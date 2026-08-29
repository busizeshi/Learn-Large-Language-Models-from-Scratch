"""
Python 内置序列类型学习 Demo

覆盖知识：
1. 容器序列 vs 扁平序列
2. 可变序列 vs 不可变序列
3. Sequence / MutableSequence 抽象基类
4. 列表推导式
5. 推导式作用域
6. 海象运算符 :=
7. map / filter 对比
8. 笛卡尔积
9. 生成器表达式
10. tuple 作为记录
11. tuple 的相对不可变性
12. tuple 的 hashable 条件
13. tuple 与 list 的内存 / 拷贝差异
14. 序列解包
15. * 收集剩余元素
16. 嵌套解包
"""

import array
import sys
from collections import deque
from collections.abc import Sequence, MutableSequence

# ==========================================================
# 1. 容器序列 vs 扁平序列
# ==========================================================

print("=" * 80)
print("1. 容器序列 vs 扁平序列")
print("=" * 80)

# ----------------------------------------------------------
# 容器序列
# ----------------------------------------------------------

person = {"name": "Alice"}

container_list = [100,"hello",person,[1, 2, 3]]

print("container_list:")
print(container_list)

print("\n每个元素的类型：")

for item in container_list:
    print(type(item), item)

# ----------------------------------------------------------
# 证明 list 保存的是对象引用
# ----------------------------------------------------------

print("\n--- list 保存的是对象引用 ---")

user = {"name": "Tom","age": 20}

users = [user]

print("修改前：")
print(users)

# 修改 user 对象
user["age"] = 21

print("修改 user 后：")
print(users)

print("user id:", id(user))
print("users[0] id:", id(users[0]))

print("是不是同一个对象：", user is users[0])

# ==========================================================
# 2. 扁平序列
# ==========================================================

print("\n" + "=" * 80)
print("2. 扁平序列")
print("=" * 80)

numbers = array.array("i",[10, 20, 30, 40])

print("array.array:")
print(numbers)

print("type:", type(numbers))

print("itemsize:", numbers.itemsize)

print("总机器数据大小：")
print(numbers.itemsize * len(numbers), "bytes")

# bytes 每个元素实际上是 0~255 的整数

data = b"ABC"

print("\nbytes:")
print(data)

print("遍历 bytes：")

for value in data:
    print(value)

# ==========================================================
# 3. 可变序列 vs 不可变序列
# ==========================================================

print("\n" + "=" * 80)
print("3. 可变序列 vs 不可变序列")
print("=" * 80)

lst = [1, 2, 3]

print("原 list：", lst)

lst[0] = 100
lst.append(4)

print("修改后：", lst)

t = (1, 2, 3)

print("\ntuple：", t)

# 下面代码会报错
#
# t[0] = 100
#
# TypeError:
# 'tuple' object does not support item assignment


text = "Python"

print("\nstr:", text)

# text[0] = "J"
#
# TypeError:
# 'str' object does not support item assignment


# ==========================================================
# 4. Sequence / MutableSequence
# ==========================================================

print("\n" + "=" * 80)
print("4. Sequence / MutableSequence")
print("=" * 80)

print("list 是 Sequence 吗？")
print(isinstance([], Sequence))

print("list 是 MutableSequence 吗？")
print(isinstance([], MutableSequence))

print()

print("tuple 是 Sequence 吗？")
print(isinstance((), Sequence))

print("tuple 是 MutableSequence 吗？")
print(isinstance((), MutableSequence))

print()

print("str 是 Sequence 吗？")
print(isinstance("", Sequence))

print("deque 是 MutableSequence 吗？")
print(isinstance(deque(), MutableSequence))

# ==========================================================
# 5. 列表推导式基础
# ==========================================================

print("\n" + "=" * 80)
print("5. 列表推导式")
print("=" * 80)

numbers = [1, 2, 3, 4, 5, 6]

squares = [n ** 2 for n in numbers]

print("平方：")
print(squares)

# 带过滤

even_squares = [n ** 2 for n in numbers if n % 2 == 0]

print("偶数平方：")
print(even_squares)

# 等价普通循环

result = []

for n in numbers:
    if n % 2 == 0:
        result.append(n ** 2)

print("普通循环：")
print(result)

# ==========================================================
# 6. 推导式作用域
# ==========================================================

print("\n" + "=" * 80)
print("6. 推导式作用域")
print("=" * 80)

x = "ABC"

codes = [ord(c) for c in x]

print(codes)

# Python 3：
#
# print(c)
#
# NameError

# ==========================================================
# 7. 海象运算符 :=
# ==========================================================

print("\n" + "=" * 80)
print("7. 海象运算符 :=")
print("=" * 80)

symbols = "ABC"

codes = [last := ord(c) for c in symbols]

print("codes:", codes)

print("last:", last)

# ==========================================================
# 8. list comprehension vs map/filter
# ==========================================================

print("\n" + "=" * 80)
print("8. 列表推导式 vs map/filter")
print("=" * 80)

words = ["python","ai","development","code","data","go"]

result1 = [len(word) ** 2 for word in words if len(word) > 3]

print("列表推导式：")
print(result1)

result2 = list(
    map(
        lambda length: length ** 2,
        filter(
            lambda length: length > 3,
            map(len, words)
        )
    )
)

print("map/filter：")
print(result2)

# ==========================================================
# 9. 笛卡尔积
# ==========================================================

print("\n" + "=" * 80)
print("9. 笛卡尔积")
print("=" * 80)

colors = ["black","white"]

sizes = ["S","M","L"]

tshirts = [(color, size) for color in colors for size in sizes]

print(tshirts)

print("\n等价普通循环：")

result = []

for color in colors:
    for size in sizes:
        result.append((color, size))

print(result)
# ==========================================================
# 10. 生成器表达式
# ==========================================================

print("\n" + "=" * 80)
print("10. 生成器表达式")
print("=" * 80)

generator = (n ** 2 for n in range(5))

print(generator)

print("类型：")
print(type(generator))

print("\n第一次 next：")
print(next(generator))

print("第二次 next：")
print(next(generator))

print("第三次 next：")
print(next(generator))
# ==========================================================
# 11. 验证生成器惰性执行
# ==========================================================

print("\n" + "=" * 80)
print("11. 验证生成器的 Lazy Evaluation")
print("=" * 80)


def calculate(n):
    print(f"正在计算 {n}")
    return n ** 2

g = (calculate(n) for n in range(5))

print("生成器已经创建")
print("----------------")
print("next(g):")
print(next(g))
print("----------------")
print("next(g):")
print(next(g))
print("----------------")
# ==========================================================
# 12. 列表 vs 生成器内存
# ==========================================================

print("\n" + "=" * 80)
print("12. list vs generator 内存")
print("=" * 80)

big_list = [ n for n in range(1_000_000)]

big_generator = (n for n in range(1_000_000))

print("list 对象大小：",sys.getsizeof(big_list))

print("generator 对象大小：",sys.getsizeof(big_generator))

# ==========================================================
# 13. 生成器用于构造 tuple / array
# ==========================================================

print("\n" + "=" * 80)
print("13. generator 用于构造其他容器")
print("=" * 80)

symbols = "$¢£¥€¤"

unicode_codes = tuple(ord(symbol) for symbol in symbols)

print("tuple:")
print(unicode_codes)

unicode_array = array.array("I", (ord(symbol) for symbol in symbols))

print("array:")
print(unicode_array)

# ==========================================================
# 14. Tuple 作为 Record
# ==========================================================

print("\n" + "=" * 80)
print("14. Tuple 作为 Record")
print("=" * 80)

city = ("Tokyo",2003,32_450,0.66,8014)

print(city)

name, year, population, change, area = city

print("城市：", name)

print("年份：", year)

print("人口：", population)

print("增长率：", change)

print("面积：", area)

# ==========================================================
# 15. Tuple 相对不可变
# ==========================================================

print("\n" + "=" * 80)
print("15. Tuple 的相对不可变性")
print("=" * 80)

data = ( "Python",[1, 2, 3])

print("原 tuple：")
print(data)

# tuple 自己不能改
#
# data[0] = "Java"
# 但它内部引用的 list 可以修改

data[1].append(4)

print("修改 list 后：")
print(data)

# ==========================================================
# 16. Tuple hashable
# ==========================================================

print("\n" + "=" * 80)
print("16. Tuple 是否可哈希")
print("=" * 80)


def is_hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError:
        return False


t1 = (10,"Python",(1, 2))

t2 = (10,"Python",[1, 2])

print("t1:",is_hashable(t1))

print("t2:",is_hashable(t2))

# t1 可以作为 dict key

cache = { t1: "cached data"}

print(cache[t1])

# ==========================================================
# 17. tuple(t) vs list(l)
# ==========================================================

print("\n" + "=" * 80)
print("17. tuple(t) vs list(l)")
print("=" * 80)

t = (1,2,3)

new_t = tuple(t)

print( "t is new_t:", t is new_t)

l = [1,2,3]

new_l = list(l)

print("l is new_l:",l is new_l)

# ==========================================================
# 18. 基础解包
# ==========================================================

print("\n" + "=" * 80)
print("18. 序列解包")
print("=" * 80)

point = (10, 20)

x, y = point

print( "x:",x)

print("y:",y)

# ==========================================================
# 19. 交换变量
# ==========================================================

print("\n" + "=" * 80)
print("19. Python 变量交换")
print("=" * 80)

a = 10
b = 20

print("交换前:",a, b)

a, b = b, a

print("交换后:",a,b)

# ==========================================================
# 20. * 函数参数展开
# ==========================================================

print("\n" + "=" * 80)
print("20. * 参数展开")
print("=" * 80)

numbers = (20,8)

result = divmod(*numbers)

print(result)

quotient, remainder = divmod(*numbers)

print(quotient,remainder)

# ==========================================================
# 21. * 收集剩余元素
# ==========================================================

print("\n" + "=" * 80)
print("21. * 收集剩余元素")
print("=" * 80)

a, b, *rest = range(5)

print("a =", a)

print("b =", b)

print("rest =", rest)

*head, tail = range(5)

print("\nhead =", head)

print("tail =", tail)

first, *middle, last = range(5)

print("\nfirst =", first)

print("middle =", middle)

print("last =", last)

# ==========================================================
# 22. 嵌套解包
# ==========================================================

print("\n" + "=" * 80)
print("22. 嵌套解包")
print("=" * 80)

metro_record = ("Tokyo", "JP",36.933,(35.689722,139.691667))

name, cc, population, (latitude,longitude) = metro_record

print("城市:",name)

print("国家:",cc)

print("人口:",population)

print("纬度:",latitude)

print("经度:",longitude)

# ==========================================================
# 23. 字面量中的 *
# ==========================================================

print("\n" + "=" * 80)
print("23. 字面量展开")
print("=" * 80)

numbers = [*range(3), 100,*(4, 5)]

print(numbers)

numbers_tuple = (*range(3),100,*(4, 5))

print(numbers_tuple)

numbers_set = {*range(3),3,*(4, 5)}

print(numbers_set)

# ==========================================================
# 24. 一个实际业务案例
# ==========================================================

print("\n" + "=" * 80)
print("24. 综合案例：处理设备采集数据")
print("=" * 80)

device_records = [("sensor-001","temperature",[23.5, 23.8, 24.1, 24.5]),("sensor-002","temperature",[25.1, 25.3, 25.6]),("sensor-003","humidity",[50, 53, 54, 55])]

# ----------------------------------------------------------
# 找出 temperature 设备
#
# 并计算平均值
# ----------------------------------------------------------

temperature_result = [(device_id,sum(values) / len(values)) for device_id, data_type, values in device_records if data_type == "temperature"]

print("温度设备平均值：")

for device_id, avg in temperature_result:
    print(device_id,avg)

# ----------------------------------------------------------
# Generator 版本
# ----------------------------------------------------------

temperature_generator = ((device_id,sum(values) / len(values)) for device_id, data_type, values in device_records if data_type == "temperature")

print("\nGenerator：")

for device_id, avg in temperature_generator:
    print(device_id,avg)
