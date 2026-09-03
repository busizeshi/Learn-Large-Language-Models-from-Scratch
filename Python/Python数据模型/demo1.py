import collections

from random import choice

Card=collections.namedtuple('Card', ['suit', 'rank'])

class FrenchDeck:
    ranks=[str(n) for n in range(2, 11)] + list('JQKA')

    suits=['spades', 'hearts', 'diamonds', 'clubs']

    def __init__(self):
        self._cards = [Card(suit, rank)
                       for suit in self.suits
                       for rank in self.ranks]
    
    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

beer_card=Card('hearts', '7')
print(beer_card)
# Card(suit='hearts', rank='7') 

deck=FrenchDeck()
print(len(deck))
# 52

# 实现了 __getitem__ 方法，可以通过索引下标轻松地从一副纸牌中抽取某一张
print(deck[0])
# Card(suit='spades', rank='2')
print(deck[1])
# Card(suit='heades', rank='3')
print(deck[-1])
# Card(suit='hearts', rank='A')
print(deck[-2])
# Card(suit='hearts', rank='K')

choice(deck)
# Card(suit='hearts', rank='J')
print(choice(deck))
# Card(suit='hearts', rank='Q')

print('-----------------'*5)

# 由于 __getitem__ 方法将操作委托给 self._cards 的 [] 运算符，因此 FrenchDeck 类将自动支持切片（slicing）操作
print(deck[:3])
# [Card(suit='spades', rank='2'), Card(suit='spades', rank='3'), Card(suit='spades', rank='4')]
print(deck[1:3])
# [Card(suit='heades', rank='3'), Card(suit='heades', rank='4')]
print(deck[1:])
# [Card(suit='heades', rank='3'), Card(suit='heades', rank='4'), Card(suit='heades', rank='5')]

print('-----------------'*5)

# 实现了特殊方法 __getitem__ 之后，这副纸牌还可以正向遍历与反向遍历
for card in deck:
    print(card)

for card in reversed(deck):
    print(card)
