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


# # import config
# import os
# import settings.config as config

# # Resources
# res = os.path.abspath("../mini-semiolog/res")+"/"
# analysis_dir = os.path.abspath(f"{res}/{config.analysis_id}/")

# # Libraries
# lib = os.path.abspath("../lib")

# # Corpus and Sentences
# corpus = f"{analysis_dir}/_corpus_/"
# sentences = f"{analysis_dir}/_sentences_/"

# # Intermediate Results
# scratch = f"{analysis_dir}/scratch/"
# vocabularies = f"{analysis_dir}/vocabularies/"
# segmentations = f"{analysis_dir}/segmentations/"
# orthogonals = f"{analysis_dir}/orthogonals/"
# # matrices = f"{analysis_dir}/matrices/"

# # Results
# segment_graphs = f"{analysis_dir}/graphs/segment_sents/"
# types = f"{analysis_dir}/types/"
# lattice_graphs = f"{analysis_dir}/graphs/lattice_types/"
# # typed_sent = f"{analysis_dir}/graphs/typed_sents/"
