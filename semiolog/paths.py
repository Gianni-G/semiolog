# FILE LOCATIONS

from pathlib import Path
from os import path

# TODO: Create all the directories referred to here

class Paths:
    def __init__(self, name) -> None:
        # TODO: configure online repository for models, with automatic download
        # self.cache = Path(path.expanduser("~/.cache/semiolog"))
        # self.online_models = "https://polybox.ethz.ch/index.php/s/bsbycrxtbeKvgEE"
        # self.models = self.cache / "models"
        
        self.models = Path("././models")
        self.semiotic = self.models / name
        self.corpus = self.models / name / "corpus"
        self.vocabulary = self.models / name / "vocabulary"
        self.paradigms = self.models / name / "paradigms"
        self.syntagmas = self.models / name / "syntagmas"
        
    def __repr__(self) -> str:
        return str(self.__dict__)