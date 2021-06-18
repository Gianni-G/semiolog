from transformers import pipeline
from .paradigm import Paradigmatizer, ParadigmChain

class Paradigmatic:
    def __init__(self,semiotic) -> None:
        self.unmasker = pipeline('fill-mask', model=semiotic.config["paradigmatic"]["model"],top_k=semiotic.config["paradigmatic"]["top_k"])
        self.paradigmatizer = Paradigmatizer()