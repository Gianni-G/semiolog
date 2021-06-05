# from collections import Counter, defaultdict
# import importlib
# import functools
# import operator
# import time
# import regex as re
# import os
# import csv
# import networkx as nx
# import random
# import graphviz as gv
# import itertools
# from scipy.sparse import csr_matrix, vstack
# import numpy as np
# import ast

from thinc.api import Config
from typing import Union, Iterable, Dict, Any
from transformers import pipeline

from .vocabulary import load_vocabulary
from . import paths
from .chain import Chain
from . import paradigm
from .text import Text


class Cenematic:
    
    def __init__(self,name) -> None:
        self.config = Config().from_disk(paths.corpora / name / "config.cfg")
        self.voc = load_vocabulary(paths.corpora / name / "vocabularies" / self.config["vocabulary"]["vocFileName"])
        self.name = name
        self.unmasker = pipeline('fill-mask', model=self.config["paradigm"]["model"],top_k=self.config["paradigm"]["top_k"])

    def __repr__(self) -> str:
        return f"Cenematic({self.name})"

    # @property
    # def unmasker(self):
    #     unmasker_f = pipeline('fill-mask', model=self.config["paradigm"]["model"],top_k=self.config["paradigm"]["top_k"])
    #     return unmasker_f
    
    def __call__(self,raw_chain):
        return Text(raw_chain,self)



    def chain(self,raw_chain):
        return Chain(raw_chain, self)

    def paradigm(self, chain):
        
        if isinstance(chain,str):
            chain = Chain(chain, self)
        return paradigm.chain_paradigm(chain,self.unmasker)
