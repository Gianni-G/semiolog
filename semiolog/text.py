from .chain import Chain
from . import paradigm

class Text:
    
    def __init__(self,raw_chain,semiotic) -> None:
        self.chain = Chain(raw_chain, semiotic)
        self.paradigm = paradigm.chain_paradigm(self.chain,semiotic)