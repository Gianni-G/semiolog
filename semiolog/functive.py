
class Functive:

    def __init__(self,sequence,span,position,semiotic) -> None:
        
        self.span = span
        self.label = sequence  
        self.position = position
        self.id = semiotic.voc.encode.get(self.label,None) # this needs ot be fixed by handling UNK!!!
        self.prob = semiotic.voc.prob.get(self.label,0)

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