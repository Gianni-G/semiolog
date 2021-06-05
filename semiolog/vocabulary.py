from collections import Counter
import csv
from typing import Union, Iterable, Dict, Any
from pathlib import Path

class Voc:
    def __init__(self,voc):
        self.len = len(voc)
        self.freq = voc
        self.freq_mass = sum(voc.values())
        self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}
    
    def __repr__(self) -> str:
        return f"Voc({self.freq})"

    def __str__(self) -> str:
        return str(self.freq)

    def __getitem__(self, item):
         return self.prob[item]

    def head(self,size=10):
        pass

    def alphabetic(self):
        pass

    def keys(self):
        pass

    def values(self):
        pass

def load_vocabulary(
    filename: Union[str,Path]
    ) -> Voc:

    """
    Load a vocabulary from existing example or local path
    RETURNS (Voc): The loaded vocabulary
    """
    
    with open(filename, "r") as f:
        csv_reader = csv.reader(f)
        voc = Counter()
        for line in csv_reader:
            voc[line[0]] = int(line[1])
        voc = dict(voc.most_common())
        
    return Voc(voc)