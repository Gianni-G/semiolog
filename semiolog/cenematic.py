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


from .vocabulary import load_vocabulary
from . import paths
from .chain import Chain



class Cenematic:
    
    def __init__(self,name) -> None:
        self.config = Config().from_disk(paths.corpora / name / "config.cfg")
        self.voc = load_vocabulary(paths.corpora / name / "vocabularies" / self.config["vocabulary"]["vocFileName"])
        self.name = name

    def __call__(self,raw_chain):
        return Chain(raw_chain, self)

    def __repr__(self) -> str:
        return f"Cenematic({self.name})"