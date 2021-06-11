from .chain import Chain
from . import paradigm
from . import ptype

class Text:
    
    def __init__(self,input_chain,semiotic) -> None:
        self.chain = Chain(input_chain, semiotic)
        # self.paradigm = paradigm.chain_paradigm(self.chain,semiotic)
        # self.type = ptype.chain_type(self.paradigm,semiotic)
