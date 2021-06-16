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
    def __init__(self, input_chain: str):
        
        self.input = input_chain
        # self.split = input_chain.split()
        # self.norm = None #normalizer(input_chain)
        # self.pre_tokens = pre_tokenize(self.norm)

        # self.semiotic = semiotic
        # self.tokenizer = Tokenizer()
        # if True: #isinstance(semiotic.config["syntagmatic"]["normalizer"], list):
            # from .tokenizer import lowercase
            # self.tokenizer.normalizer = lowercase
        # self.tokenizer.normalizer = 
        # self.tokenizer.pre_tokenizer = 
        # self.tokenizer.processor
        # self.tokenizer.post_processor
        # self.tokenizer.encoder
        # self.tokenizer.decoder




        # self.tokens = tokenize(self.pre_tokens, self.tokenizer)

        # self.len = len(self.tokens)

        # self.labels = [token.label for token in self.tokens]
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

    # def mask(self,n):
    #     """
    #     Outputs a new list with the nth token of the chain replaced with the "[MASK]" token
    #     """
    #     if isinstance(n,int):
    #         n = [n]
    #     masked_chain = [token if i not in n else Functive("[MASK]",token.span,token.position,self.model) for i,token in enumerate(self)]
    #     return masked_chain




