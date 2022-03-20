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
        self.split_norm = [self.semiotic.syntagmatic.tokenizer.normalizer.normalize_str(s) for s in self.split]
        self.split_norm = [t for t in self.split_norm if t!='']

        # self.norm = None
        # self.pre_tokens = None
        # self.processor = None

        # if semiotic.config.syntagmatic.SLG_processor != None:
        #     self.SLG_input = self.semiotic.syntagmatic.SLG_tokenizer(self.input, self.semiotic)
        # else:
        #     self.SLG_input = self.input

        # hf_output = semiotic.syntagmatic.tokenizer.encode(self.SLG_input)
        
        hf_output = semiotic.syntagmatic.tokenizer.encode(self.input)

        #TODO: Using HF tokenizers, the tokens for trees are yet to be done, and it's not clear yet how tree tokens could work. Also, the "offsets" or "span" for inputs without spaces have to be double checked

        self.tree_tokens = [Functive(segment,span,position,ids,semiotic) for segment,span,position,ids in zip(hf_output.tokens,hf_output.offsets,hf_output.word_ids,hf_output.ids)]

        self.tokens = sorted([token for token in self.tree_tokens if token.position != None], key= lambda x: x.position)

        chain = "".join(self.split_norm)
        tree_root = Functive(chain, (0, len(chain)), None, None, semiotic)
        tree_root.children = self.tokens

        for token in self.tokens:
            token.head = tree_root

        self.len = len(self.tokens)
        self.labels = [token.label for token in self.tokens]


        # self.segmented = " ".join(self.labels)

    @property
    def nodes_split(self):
        nodes_list = []
        for i in range(len(self.split_norm)):
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
        Outputs a string with the nth label(s) of the chain replaced with the mask token. n can be an integer or a list of integers.
        """
        if isinstance(n,int):
            n = [n]

        assert max(n)<self.len, f"SLG: The mask position ({max(n)}) is bigger than the length of the chain ({self.len})."

        mask_token = self.semiotic.config.vocabulary.mask_token

        masked_chain = " ".join([t if i not in n else mask_token for i,t in enumerate(self.labels)])
        
        return masked_chain   

    def mask_tokens(self,n):
        """
        Outputs a new list with the nth token(s) of the chain replaced with the mask token. n can be an integer or a list of integers.
        """
        if isinstance(n,int):
            n = [n]

        assert max(n)<self.len, f"SLG: The mask position ({max(n)}) is bigger than the length of the chain ({self.len})."

        mask_token = self.semiotic.config.vocabulary.mask_token
        # This id for mask_token depends on the tokenizer being able to include the mask_token, which, due to a HF bug, happens at the end (hence the high id). Double-check when the bug has been dealt with
        mask_token_id = self.semiotic.syntagmatic.tokenizer.token_to_id(mask_token)

        masked_chain = [token if i not in n else Functive(mask_token,token.span,token.position,mask_token_id,self.semiotic) for i,token in enumerate(self.tokens)]
        
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






