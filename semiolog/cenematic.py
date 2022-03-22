import os
from psutil import cpu_count

from .paths import Paths
from .config import Config
from .corpus import Corpus
from .vocabulary import Vocabulary #, nGram
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

        # TODO: configure online repository for models, with automatic download
        
        if not empty:
            
            if os.path.isdir(self.paths.semiotic) and not config_only:
                self.corpus.from_file()
                if vocab:
                    self.vocab.from_file()
            
            else:
                createQ = input(f"No existing model correspond to the name {self.name}.\n Do you want to create it? (Y/N)")
                if createQ == "Y":
                    os.mkdir(self.paths.semiotic)
                    os.mkdir(self.paths.corpus)
                    os.mkdir(self.paths.corpus / "original")
                    os.mkdir(self.paths.vocabulary)
                    os.mkdir(self.paths.paradigms)
                    self.config.save()
                    print(f"\nFolder for model '{self.name}' created in {self.paths.models.absolute()}")
                
        self.syntagmatic = Syntagmatic(self)
        # self.ng2 = nGram()
        self.paradigmatic = Paradigmatic(self)
        # self.typing = Typing(self)
        
        # # Load universal dependencies (ud) and constituency parsing (cp) models
        # self.ud = spacy.load(self.config["evaluation"]["ud_model"])
        # self.cp = spacy.load(self.config["evaluation"]["cp_model"])
        # self.cp.add_pipe("benepar", config={"model": "benepar_en3"})

    def __repr__(self) -> str:
        return f"Cenematic({self.name})"

    def __call__(self,input_chain):
        return Text(input_chain,self)