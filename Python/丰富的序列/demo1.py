# 列表推导式

squares = [n * n for n in range(1, 6)]
print(squares)
# [1, 4, 9, 16, 25]

even_squares = [n for n in range(2, 6) if n % 2 == 0]
print(even_squares)
# [2, 4]

labels = ["偶数" if n % 2 == 0 else "奇数" for n in range(5)]
print(labels)
# ['偶数', '奇数', '偶数', '奇数', '偶数']

pairs = [(x, y) for x in "ab" for y in range(2)]
print(pairs)
# [('a', 0), ('a', 1), ('b', 0), ('b', 1)]

print("-----------------" * 5)

# 生成器表达式
gen = (n * n for n in range(1, 6))
print(gen)
# <generator object <genexpr> at 0x00000152072635E0> 只有当需要时才会计算

print(next(gen))
# 1

print(next(gen))
# 4

print(list(gen))
# [9, 16, 25]  打印剩余的元素

# 惰性求值，按需产出，不占用大内存
import sys

lst = [n * n for n in range(10000000)]
gen = (n * n for n in range(10000000))

print(sys.getsizeof(lst)/1024/1024)
# 89095160
print(sys.getsizeof(gen))
# 200

print("-----------------" * 5)

# 海象表达式
codes=[last:=ord(c) for c in ["a","b","c","d","e"]]
print(codes)
# [97, 98, 99, 100, 101]
print(last)
# 101