from .syntagmatic import Chain
from .paradigmatic import ParadigmChain
from . import ptype

class Text:
    
    def __init__(self,input_chain,semiotic) -> None:
        self.chain = Chain(input_chain,semiotic)
        semiotic.syntagmatic.tokenizer(self.chain)
        # semiotic.paradigmatic(self.chain)

        # self.paradigmatic = ParadigmChain(self.chain,semiotic)
        # self.type = ptype.chain_type(self.paradigm,semiotic)
