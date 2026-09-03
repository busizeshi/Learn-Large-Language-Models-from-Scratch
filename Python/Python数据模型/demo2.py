import math

class Vector:
    def __init__(self,x=0,y=0) -> None:
        self.x=x
        self.y=y
    
    def __repr__(self) -> str:
        return f'Vector({self.x!r},{self.y!r})'

    def __abs__(self) -> float:
        return math.hypot(self.x,self.y)

    def __bool__(self) -> bool:
        return bool(abs(self))

    def __add__(self, other: 'Vector') -> 'Vector':
        x=self.x+other.x
        y=self.y+other.y
        return Vector(x,y)
    
    def __mul__(self, scalar: float) -> 'Vector':
        x=self.x*scalar
        y=self.y*scalar
        return Vector(x,y)

v1=Vector(2,4)
v2=Vector(2,1)
print(v1+v2)
# Vector(4,5)
v=Vector(3,4)
print(abs(v))
# 5.0
print(v*3)
# Vector(9,12)
print(abs(v*3))
# 15.0
print(bool(v))
# True
print(repr(v))
# Vector(3, 4)
v3=Vector(0,0)
print(bool(v3))
# False