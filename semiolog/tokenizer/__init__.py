from . import normalizers
from . import pre_tokenizers
from . import processors
from . import post_processors

class Tokenizer:
    """
    """

    def __init__(self,config,semiotic) -> None:
        # TODO: There must be a cleverer way of skipping steps in the pipeline, other than attributing the base class of the step (ex: "PreTokenizer" for pre-tokenizers). Maybe using the keyword "disable"
        if isinstance(config['normalizer'],list):
            self.normalizer = normalizers.Sequence(config['normalizer'])
        else:
            self.normalizer = eval(f"normalizers.{config['normalizer']}()")
        self.pre_tokenizer = eval(f"pre_tokenizers.{config['pre_tokenizer']}()")
        self.is_pretokenized = False if self.pre_tokenizer == pre_tokenizers.PreTokenizer else True
        self.processor = eval(f"processors.{config['processor']}()")
        self.post_processor = eval(f"post_processors.{config['post_processor']}()")

    def encoder(self,input_string):
        pass

    def decoder(self,input_string):
        pass
    
    # def __call__(self,chain,semiotic):
    #     chain.norm = self.normalizer.normalize(chain.input)
    #     chain.pre_tokens = self.pre_tokenizer.pre_tokenize(chain.norm)
    #     chain.processor = self.processor.process(chain.pre_tokens, semiotic, is_pretokenized = self.is_pretokenized)
    #     chain.tokens = self.post_processor.post_process(chain.processor)