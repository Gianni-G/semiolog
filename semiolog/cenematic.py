import os
from psutil import cpu_count

from .paths import Paths
from .config import Config
from .corpus import Corpus
from .vocabulary import Vocabulary, nGram
from .syntagmatic import Syntagmatic
from .paradigmatic import Paradigmatic
from .typing import Typing
from .text import Text

class Cenematic:
    
    def __init__(
        self,
        name,
        empty=False,
        config_only = False,
        requested_cpu = None,
        vocab = True,
        ) -> None:

        self.name = name
        self.paths = Paths(self.name)
        
        self.config = Config(self)
        if empty == False and os.path.isdir(self.paths.semiotic):
            self.config.from_file()            
        self.config.system.cpu_count = cpu_count(logical = False) if requested_cpu == None else requested_cpu

        self.corpus = Corpus(self)
        self.vocab = Vocabulary(self)
        
        self.syntagmatic = Syntagmatic(self)

        # self.ng2 = nGram()
        
        self.paradigmatic = Paradigmatic(self)
        self.typing = Typing(self)

        # TODO: configure online repository for models, with automatic download
        
        # TODO: Take a look at the loading order, since I moved the config from file up
        if empty == False and os.path.isdir(self.paths.semiotic):
            # self.config.from_file()
            # self.config.system.cpu_count = cpu_count(logical = False) if requested_cpu == None else requested_cpu
            if not config_only:
                self.corpus.from_file()
                if vocab:
                    self.vocab.from_file()
                self.syntagmatic = Syntagmatic(self)

                # self.ng2 = nGram(paths.examples / name / "ngrams" / self.config["vocabulary"]["nGramFileName"])

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