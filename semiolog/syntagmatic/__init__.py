# Chain and Tree are needed elsewhere
from .chain import Chain
from .tree import Tree

from os import path
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

# from ...util import if_none_disable

# TODO: unable to import this from ...util
def if_none_disable(x):
    if x == None:
        x = "disable"
    return x

class Syntagmatic:
    def __init__(self,semiotic) -> None:
        
        self.config = semiotic.config.syntagmatic
        self.config_vocab = semiotic.config.vocabulary
        self.tokenizer_path = str(semiotic.paths.vocabulary.joinpath("tokenizer.json"))

        # Load HF Tokenizer model (from "processor" config)

        if self.config.from_file and path.exists(self.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            self.tokenizer = Tokenizer(
                eval(
                    f"models.{self.config.processor}(vocab = {semiotic.vocab.encode},unk_token = '{self.config_vocab.unk_token}')"
                    )
                    )
            
            # Load HF normalizer
            if isinstance(self.config.normalizer,list):
                self.tokenizer.normalizer = normalizers.Sequence(
                    [eval(f"normalizers.{norm}()") for norm in self.config.normalizer]
                    )
            else:
                self.tokenizer.normalizer = eval(f"normalizers.{if_none_disable(self.config.normalizer)}()")

            # Load HF pre-tokenizer
            if isinstance(self.config.pre_tokenizer,list):
                self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                    [eval(f"normalizers.{norm}()") for norm in self.config.pre_tokenizer]
                    )
            else:
                self.tokenizer.pre_tokenizer = eval(f"pre_tokenizers.{if_none_disable(self.config.pre_tokenizer)}()")
            
            # # Possible post_processor in case needed.

            # cls_token_id = tokenizer.token_to_id(self.config_vocab.cls_token)
            # sep_token_id = tokenizer.token_to_id(self.config_vocab.sep_token)

            # tokenizer.post_processor = processors.TemplateProcessing(
            #     single=f"[CLS]:0 $A:0 [SEP]:0",
            #     pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            #     special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
            # )
    
        self.bert_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object = self.tokenizer,
            # tokenizer_file=str(semiotic.vocab.path.joinpath("tokenizer.json")), # You can load from the tokenizer file, alternatively

            unk_token= self.config_vocab.unk_token,
            pad_token= self.config_vocab.pad_token,
            cls_token= self.config_vocab.cls_token,
            sep_token= self.config_vocab.sep_token,
            mask_token= self.config_vocab.mask_token,

            model_max_length = self.config.model_max_length,

        )

    def save_tokenizer(self):
        self.tokenizer.save(self.tokenizer_path)