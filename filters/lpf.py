class LPF(object):
    def __init__(self, x , alph):
        self.x = x
        self.alph = alph
    
    def __call__(self, u):
        x = self.x
        alph = self.alph
        self.x = u * (1.0-alph) + x * (alph)        
        return self.x