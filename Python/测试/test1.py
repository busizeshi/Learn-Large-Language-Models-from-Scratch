ranks=[str(n) for n in range(2, 11)]
# ['2', '3', '4', '5', '6', '7', '8', '9', '10']
ranks=ranks+list('JQKA')

# list() 构造函数接受任何可迭代对象，把其中每个元素收集起来放进一个新列表
print(list('JQKA'))
# ['J', 'Q', 'K', 'A']

print(ranks)