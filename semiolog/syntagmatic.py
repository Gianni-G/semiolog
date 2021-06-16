import networkx as nx
import graphviz as gv
from thinc.api import Config


from . import util
from . import util_g
from .functive import Functive


class ChainIterator:
    ''' Iterator class '''
    def __init__(self, chain):
        self.chain = chain
        # member variable to keep track of current index
        self.index = 0
    def __next__(self):
        if self.index < self.chain.len:
            result = self.chain[self.index]
            self.index +=1
            return result
        raise StopIteration

class Chain:
    def __init__(self, input_chain: str,semiotic):
        
        self.semiotic = semiotic
        self.input = input_chain
        self.split = input_chain.split()

        self.norm = None
        self.pre_tokens = None
        self.processor = None
        self.tokens = None

        self.len = None
        self.labels = None

    def mask(self,n):
        """
        Outputs a new list with the nth token(s) of the chain replaced with the "[MASK]" token. n can be an integer or a list of integers.
        """
        if isinstance(n,int):
            n = [n]
        masked_chain = [token if i not in n else Functive("[MASK]",token.span,token.position,self.semiotic) for i,token in enumerate(self.tokens)]
        return masked_chain   






        # self.segmented = " ".join(self.labels)

        # self.nodes_list = []
        # for i in range(len(self.split)):
        #     start_i = len("".join(self.split[:i]))
        #     end_i = start_i + len(self.split[i])
        #     self.nodes_list.append((self.split[i], (start_i, end_i)))
            
        # self.nodes = set(self.nodes_list)
        

    def __repr__(self) -> str:
        return f"Chain({self.input})"

    # def __str__(self) -> str:
    #     return self.segmented

    def __iter__(self):
       ''' Returns the Iterator object '''
       return ChainIterator(self)

    # def __getitem__(self, index:str):
    #     return self.tokens[index]






