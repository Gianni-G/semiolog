from .chain import Chain
from .paradigm import ParadigmChain
from . import ptype

class Text:
    
    def __init__(self,input_chain,semiotic) -> None:
        self.chain = Chain(input_chain, semiotic)
        self.paradigm = ParadigmChain(self.chain,semiotic)
        # self.type = ptype.chain_type(self.paradigm,semiotic)
