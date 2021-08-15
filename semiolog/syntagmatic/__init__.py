from .tokenizer import Tokenizer
from .chain import Chain
from .tree import Tree

class Syntagmatic:
    def __init__(self,semiotic) -> None:
        
        self.config = semiotic.config.syntagmatic
        # self.tokenizer = Tokenizer(self.config)
    
    @property
    def tokenizer(self):
        return Tokenizer(self.config)
        