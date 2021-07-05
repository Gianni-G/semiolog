from transformers import pipeline
from .paradigm import Paradigmatizer, ParadigmChain

class Paradigmatic:
    def __init__(self,semiotic) -> None:
        self.config = semiotic.config.paradigmatic
        # self.unmasker = pipeline('fill-mask', model = self.config.model,top_k = self.config.top_k)
        self.paradigmatizer = Paradigmatizer()