
class Functive:

    def __init__(self,label,span,position,semiotic) -> None:
        
        self.span = span
        self.label = label  
        self.position = position
        self.id = semiotic.vocab.encode.get(self.label,None) # this needs ot be fixed by handling UNK!!!
        self.prob = semiotic.vocab.prob.get(self.label,0)

        # self.parad
        # self.parad_t
        # self.parad_t_soft
        # self.prob_in_parad

        # self.f_type
        # self.type

        # self.head
        # self.children
        # self.subtree
        
    def __repr__(self) -> str:
        return f"Functive({self.label},{self.span})"
    
    def __str__(self) -> str:
        return str(self.label)