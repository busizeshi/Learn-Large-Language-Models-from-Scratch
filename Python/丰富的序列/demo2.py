# map&filter
words = ["apple", "fig", "binana", "kiwi"]

# map实现逐项转换
upper = map(str.upper, words)
print(upper)
# <map object at 0x000002546BD32050>    惰性：还没计算
print(list(upper))
# ['APPLE', 'FIG', 'BINANA', 'KIWI']

# filter实现逐项判定
long = filter(lambda w: len(w) > 3, words)
print(long)
# <filter object at 0x000001F1370422F0>
print(list(long))
# ['apple', 'binana', 'kiwi']
