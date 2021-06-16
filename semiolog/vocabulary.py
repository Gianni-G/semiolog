from collections import Counter
import csv
from typing import Union, Iterable, Dict, Any
from pathlib import Path

class Vocabulary:
    def __init__(self,filename = None,special_tokens = None): # TODO: Handle special tokens
        if filename != None:
                
            with open(filename, "r") as f:
                csv_reader = csv.reader(f)
                voc = Counter()
                for line in csv_reader:
                    voc[line[0]] = int(line[1])
                voc = dict(voc.most_common())

            self.len = len(voc)
            self.freq = voc
            self.freq_mass = sum(voc.values())
            self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

            self.encode = {k:i for i,(k,v) in enumerate(voc.items())}
            self.decode = {i:k for k,i in self.encode.items()}
        else:
            pass


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