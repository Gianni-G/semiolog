from . import normalizers
from . import pre_tokenizers
from . import processors
from . import post_processors

class Tokenizer:
    """
    """

    def __init__(self,semiotic) -> None:

        self.config = semiotic.config.syntagmatic
        
        if isinstance(self.config.normalizer,list):
            self.normalizer = normalizers.Sequence(self.config.normalizer)
        else:
            self.normalizer = eval(f"normalizers.{self.config.normalizer}()")
        
        self.pre_tokenizer = eval(f"pre_tokenizers.{self.config.pre_tokenizer}()")
        self.is_pretokenized = False if self.pre_tokenizer == pre_tokenizers.PreTokenizer else True
        self.processor = eval(f"processors.{self.config.processor}()")
        self.post_processor = eval(f"post_processors.{self.config.post_processor}()")

    def encoder(self,input_string):
        pass

    def decoder(self,input_string):
        pass

    def __call__(self,chain):
        chain.norm = self.normalizer.normalize(chain.input)
        chain.pre_tokens = self.pre_tokenizer.pre_tokenize(chain.norm)
        chain.processor = self.processor.process(chain.pre_tokens, chain.semiotic, is_pretokenized = self.is_pretokenized)
        chain.tree_tokens = self.post_processor.post_process(chain.processor)
        chain.tokens = [token for token in chain.tree_tokens if token.position != None]

        chain.len = len(chain.tokens)
        chain.labels = [token.label for token in chain.tokens]
        chain.probs = [token.prob for token in chain.tokens]
        pass