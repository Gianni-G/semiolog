from .tokenizer import Tokenizer
from .chain import Chain

class Syntagmatic:
    def __init__(self,semiotic) -> None:
        self.tokenizer = Tokenizer(semiotic)
