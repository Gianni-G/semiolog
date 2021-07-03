from thinc.api import Config
import os

from .paths import Paths
from .vocabulary import Vocabulary, nGram
from .syntagmatic import Syntagmatic
from .paradigmatic import Paradigmatic
from .typing import Typing
from .text import Text
from .config import Config


class Cenematic:
    
    def __init__(self,name) -> None:

        self.name = name
        self.paths = Paths(self.name)

        if not os.path.isdir(self.paths.semiotic):
            self.config = Config(self)
            # self.vocab = Vocabulary()
            # self.ng2 = nGram()

            # self.syntagmatic = Syntagmatic(self)
            # self.paradigmatic = Paradigmatic(self)
            # self.typing = Typing(self)

        else:
            self.config = Config().from_disk(paths.examples / name / "config.cfg")
            self.vocab = Vocabulary(paths.examples / name / "vocabularies" / self.config["vocabulary"]["vocFileName"])
            self.ng2 = nGram(paths.examples / name / "ngrams" / self.config["vocabulary"]["nGramFileName"])

            self.syntagmatic = Syntagmatic(self)
            self.paradigmatic = Paradigmatic(self)
            self.typing = Typing(self)
        

        # # Load universal dependencies (ud) and constituency parsing (cp) models
        # self.ud = spacy.load(self.config["evaluation"]["ud_model"])
        # self.cp = spacy.load(self.config["evaluation"]["cp_model"])
        # self.cp.add_pipe("benepar", config={"model": "benepar_en3"})

    def __repr__(self) -> str:
        return f"Cenematic({self.name})"

    def __call__(self,input_chain):
        return Text(input_chain,self)


    def test_sents(self,filename = None):
    
        if filename == None:
            filename = self.config["general"]["testSentences"]

        with open(paths.examples / self.name / "_sentences_" / f"{filename}.txt", "r") as f:
            sents = []
            for line in f.readlines():
                sents.append(line.rstrip())
        return sents
