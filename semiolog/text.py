from .syntagmatic import Chain
from .paradigmatic import ParadigmChain
from . import ptype

class Text:
    
    def __init__(self,input_chain,semiotic) -> None:
        self.chain = Chain(input_chain,semiotic)
        # semiotic.tokenizer(self.chain,semiotic)

        # self.syntagmatic = Chain(input_chain,semiotic)
        # self.paradigmatic = ParadigmChain(self.chain,semiotic)
        # self.type = ptype.chain_type(self.paradigm,semiotic)
