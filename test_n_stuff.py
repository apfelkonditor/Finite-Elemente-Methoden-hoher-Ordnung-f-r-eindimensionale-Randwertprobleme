class real():
    
    def __init__(self, a) -> None:
        self.a = a

    def __mul__(self,other):
        return real(self.a * other.a)

    def __rmul__(self,other):
        return real(other * self.a * 2)

    
class complex(real):

    def __init__(self, a, b) -> None:
        super().__init__(a)
        self.b = b
    
    def __mul__(self, other):
        return real(self.atr2 + other.atr2)


    
    def __rmul__(self, other):
        return real(self.atr2 + other.atr2)
    
class Testfunction():
    pass


