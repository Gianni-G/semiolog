# FILE LOCATIONS

from pathlib import Path
from os import path, makedirs

# TODO: Create all the directories referred to here

class Paths:
    def __init__(self, name) -> None:
        self.cache = Path(path.expanduser("~/.cache/semiolog"))
        self.online_models = "https://polybox.ethz.ch/index.php/s/bsbycrxtbeKvgEE"
            
        self.models = self.cache / "models"
        print('bla')
        if not path.isdir(self.models):
            makedirs(self.models)
        print("bla")
        self.semiotic = self.models / name
        self.corpus = self.models / name / "corpus"
        self.vocabulary = self.models / name / "vocabulary"

        
    def __repr__(self) -> str:
        return str(self.__dict__)