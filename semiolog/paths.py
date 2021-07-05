# FILE LOCATIONS

from pathlib import Path

# TODO: Create all the directories referred to here

class Paths:
    def __init__(self, name) -> None:
        self.models = Path("././models")
        self.semiotic = self.models / name
        self.corpus = self.models / name / "corpus"
        self.vocabulary = self.models / name / "vocabulary"
        
    def __repr__(self) -> str:
        return str(self.__dict__)