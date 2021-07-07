from ..functive import Functive


class ChainIterator:
    ''' Iterator class '''
    def __init__(self, chain):
        self.iterable = chain.tokens
        self.len = chain.len
        # member variable to keep track of current index
        self.index = 0
    def __next__(self):
        if self.index < self.len:
            result = self.iterable[self.index]
            self.index +=1
            return result
        raise StopIteration

class Chain:
    def __init__(self, input_chain: str,semiotic):
        
        self.semiotic = semiotic
        self.input = input_chain
        self.split = input_chain.split()
        self.split_norm = [self.semiotic.syntagmatic.tokenizer.normalizer.normalize(s) for s in self.split]

        self.norm = None
        self.pre_tokens = None
        self.processor = None
        self.tokens = None

        self.len = None
        self.labels = None

        # self.segmented = " ".join(self.labels)

    @property
    def nodes_split(self):
        nodes_list = []
        for i in range(len(self.split)):
            start_i = len("".join(self.split_norm[:i]))
            end_i = start_i + len(self.split_norm[i])
            nodes_list.append((
                self.split_norm[i],
                (start_i, end_i)
                ))
            
        return set(nodes_list)
    
    @property
    def nodes(self):
        return [(token.label, token.span) for token in self.tokens]
        
    def mask(self,n):
        """
        Outputs a new list with the nth token(s) of the chain replaced with the "[MASK]" token. n can be an integer or a list of integers.
        """
        if isinstance(n,int):
            n = [n]
        masked_chain = [token if i not in n else Functive("[MASK]",token.span,token.position,self.semiotic) for i,token in enumerate(self.tokens)]
        return masked_chain   







        

    def __repr__(self) -> str:
        return f"Chain({self.input})"

    # def __str__(self) -> str:
    #     return self.segmented

    def __iter__(self):
       ''' Returns the Iterator object '''
       return ChainIterator(self)

    # def __getitem__(self, index:str):
    #     return self.tokens[index]






