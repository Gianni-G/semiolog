from collections import Counter
import csv
from typing import Union, Iterable, Dict, Any
from pathlib import Path
from tqdm.notebook import trange

class Vocabulary:
    def __init__(self,filename = None,special_tokens = None): # TODO: Handle special tokens
        if filename != None:
                
            with open(filename, "r") as f:
                csv_reader = csv.reader(f)
                voc = Counter()
                for line in csv_reader:
                    voc[line[0]] = int(line[1])
                voc = dict(voc.most_common())

            self.filename = filename
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


class nGram(Vocabulary):
    def __init__(self, filename = None, special_tokens = None):

        if filename != None:
                
            with open(filename, "r") as f:
                csv_reader = csv.reader(f)
                voc = Counter()
                for line in csv_reader:
                    voc[tuple(line[:2])] = int(line[-1])
                voc = dict(voc.most_common())

            self.filename = filename
            self.len = len(voc)
            self.freq = voc
            self.freq_mass = sum(voc.values())
            self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

            self.encode = {k:i for i,(k,v) in enumerate(voc.items())}
            self.decode = {i:k for k,i in self.encode.items()}
        else:
            pass

    def __repr__(self) -> str:
        return f"nGram({self.freq})"

    def __str__(self) -> str:
        return str(self.freq)

    def __getitem__(self, item):
         return self.prob[item]
    

from collections import Counter
import regex as re
from tqdm.notebook import trange
from .syntagmatic import tokenizer

# from semiolog import util_g
# import functools
# import time
# import operator

#TODO: This must be parallelizable, but no gain of efficiency so far
def find_best_pair(chain_spaced):
    pre_units = chain_spaced.split()
    pre_units_pairs = zip(pre_units, pre_units[1:])
    pairs = Counter(pre_units_pairs)
    return pairs.most_common()[0]

def agglutinate_chain(pair, chain_spaced):
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_chain = p.sub("".join(pair), chain_spaced)
    return new_chain

def build(
    corpus,
    vocab_size,
    special_tokens = None,
    ):
    
    normalizer = tokenizer.normalizers.Sequence(["Lowercase","StripPunctuation","StripWhitespaces"])
    
    chain = normalizer.normalize("".join(corpus.train))
    
    chain = " ".join(chain)
    
    vocabulary = Counter(chain.split()).most_common()
    
    special_tokens_len = 0 if special_tokens == None else len(special_tokens)
    voc_len = len(vocabulary) + special_tokens_len
    pair = vocabulary[0][0]
    
    merges = []
    t = trange(vocab_size - voc_len, leave=True)
    for i in t:
        t.set_description(f"Pair: {pair})\t")
        t.refresh()

        pair = find_best_pair(chain)
        chain = agglutinate_chain(pair[0], chain)
        merges.append(" ".join(pair[0]))
    
    vocabulary = Counter(chain.split()).most_common()
        
    if special_tokens != None:
        vocabulary = vocabulary + [(token,0) for token in special_tokens]
        
    return (vocabulary,merges)