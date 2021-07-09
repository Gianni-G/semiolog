
from . import normalizers
from . import pre_tokenizers
from . import processors
from . import post_processors

# from ...util import if_none_disable

# TODO: unable to import this from ...util
def if_none_disable(x):
    if x == None:
        x = "disable"
    return x

class Tokenizer:
    """
    """

    def __init__(self,config) -> None:

        # self.config = {k:v if v!= None else "disable" for k,v in config.__dict__.items()}
        self.config = config
        

        # TODO: There should be a better way to convert config "None" into "disable"
        
        if isinstance(self.config.normalizer,list):
            self.normalizer = normalizers.Sequence(self.config.normalizer)
        else:
            self.normalizer = eval(f"normalizers.{if_none_disable(self.config.normalizer)}()")
        
        self.pre_tokenizer = eval(f"pre_tokenizers.{if_none_disable(self.config.pre_tokenizer)}()")
        self.is_pretokenized = False if self.pre_tokenizer == pre_tokenizers.PreTokenizer else True
        self.processor = eval(f"processors.{if_none_disable(self.config.processor)}()")
        self.post_processor = eval(f"post_processors.{if_none_disable(self.config.post_processor)}()")

    def encoder(self,input_string):
        pass

    def decoder(self,input_string):
        pass

    def __call__(self,chain):
        chain.norm = self.normalizer.normalize(chain.input)
        chain.pre_tokens = self.pre_tokenizer.pre_tokenize(chain.norm)
        chain.processor = self.processor.process(chain.pre_tokens, chain.semiotic, is_pretokenized = self.is_pretokenized)
        chain.tree_tokens = self.post_processor.post_process(chain.processor)
        chain.tokens = sorted([token for token in chain.tree_tokens if token.position != None], key= lambda x: x.position)

        chain.len = len(chain.tokens)
        chain.labels = [token.label for token in chain.tokens]
        chain.probs = [token.prob for token in chain.tokens]
        pass