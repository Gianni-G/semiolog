from thinc.api import Config
from transformers import pipeline

from . import paths
from .vocabulary import Vocabulary
from .syntagmatic import Syntagmatic
from .text import Text


class Cenematic:
    
    def __init__(self,name) -> None:
        self.name = name
        self.config = Config().from_disk(paths.corpora / name / "config.cfg")
        self.vocab = Vocabulary(paths.corpora / name / "vocabularies" / self.config["vocabulary"]["vocFileName"])

        self.syntagmatic = Syntagmatic(self)
        

        # self.paradigmatic = 
        

        # # Load universal dependencies (ud) and constituency parsing (cp) models
        # self.ud = spacy.load(self.config["evaluation"]["ud_model"])
        # self.cp = spacy.load(self.config["evaluation"]["cp_model"])
        # self.cp.add_pipe("benepar", config={"model": "benepar_en3"})

        # Load transformers "fill-mask" task

        # self.unmasker = pipeline('fill-mask', model=self.config["paradigmatic"]["model"],top_k=self.config["paradigmatic"]["top_k"])



    def __repr__(self) -> str:
        return f"Cenematic({self.name})"

    
    def __call__(self,input_chain):
        return Text(input_chain,self)

    # def syntagmatic(self,input_chain):
    #     return Chain(input_chain, self)

    # def paradigmatic(self, chain):
        
    #     if isinstance(chain,str):
    #         chain = Chain(chain, self)
    #     return ParadigmChain(chain,self.unmasker)

    def test_sents(self,filename = None):
    
        if filename == None:
            filename = self.config["general"]["testSentences"]

        with open(paths.corpora / self.name / "_sentences_" / f"{filename}.txt", "r") as f:
            sents = []
            for line in f.readlines():
                sents.append(line.rstrip())
        return sents
