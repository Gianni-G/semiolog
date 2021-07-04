from .tokenizer import Tokenizer
from .chain import Chain
from .tree import Tree

class Syntagmatic:
    def __init__(self,semiotic) -> None:
        
        self.semiotic = semiotic
        self.tokenizer = Tokenizer(self.semiotic)
        