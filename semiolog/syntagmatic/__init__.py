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

    Regex,
    NormalizedString,
    PreTokenizedString,
)

from .tokenizer import SequenceSLG, PMI_SLG, NormalizeSLG, Normalize_w_spacesSLG

from transformers import PreTrainedTokenizerFast
import datasets

class Syntagmatic:
    def __init__(self,semiotic) -> None:
        
        self.config = semiotic.config.syntagmatic
        self.config_vocab = semiotic.config.vocabulary
        self.dataset = semiotic.corpus.dataset
        self.split_rate = semiotic.config.corpus.split_rate
        self.path = semiotic.paths.syntagmas
        
        self.tokenizer_path = str(semiotic.paths.vocabulary.joinpath("tokenizer.json"))

        # Load HF Tokenizer model (from "processor" config)

        if self.config.from_file and path.exists(self.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
            print(f"SLG [I]: Tokenizer loaded from file")
        else:
            allowed_processors = {"BPE", "WordLevel"}
            if self.config.processor not in allowed_processors:
                raise Exception(f"SLG [E]: Processor provided in config.syntagmatic should be among the following: {allowed_processors}. The processor provided was: {self.config.processor}")
            if self.config.processor == "WordLevel":
                self.tokenizer = Tokenizer(
                    models.WordLevel(
                        vocab = semiotic.vocab.encode,
                        unk_token = self.config_vocab.unk_token
                        )
                        )
            elif self.config.processor == "BPE":
                
                # By the way the vocab is built in SLG, some early merged terms might not appear in the final vocabulary. HF BPE needs them to be there, so they are added here in case it is needed.

                max_vocab_encode = max(semiotic.vocab.encode.values())
                missing_merges = [(k,max_vocab_encode+i+1) for i,k in enumerate({"".join(m) for m in semiotic.vocab.merges}-semiotic.vocab.encode.keys())]

                if missing_merges != []:
                    for k,i in missing_merges:
                        semiotic.vocab.encode[k] = i
                    print(f"SLG [I]: The following merged terms have been added to the vocabulary: {missing_merges}")

                self.tokenizer = Tokenizer(
                    models.BPE(
                        vocab = semiotic.vocab.encode,
                        merges = semiotic.vocab.merges,
                        unk_token = self.config_vocab.unk_token
                        )
                        )
            
            # Load HF normalizer
            # The elif condition on the string SLG is sort of a hack (needed due to non standard declaration of custom normalizer). There should be a more elegant way
            if self.config.normalizer != None:
                if isinstance(self.config.normalizer,list):
                    self.tokenizer.normalizer = normalizers.Sequence(
                        [eval(f"normalizers.{norm}()") for norm in self.config.normalizer]
                        )
                elif self.config.normalizer[-3:] == "SLG":
                    self.tokenizer.normalizer = eval(f"{self.config.normalizer}")
                else:
                    self.tokenizer.normalizer = eval(f"normalizers.{self.config.normalizer}()")

            # Load HF pre-tokenizer
            # The elif condition on the string SLG is sort of a hack (needed due to non standard declaration of custom pre-tokenizer). There should be a more elegant way
            if self.config.pre_tokenizer != None:
                if isinstance(self.config.pre_tokenizer,list):
                    self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                        [eval(f"pre_tokenizer.{pretok}()") for pretok in self.config.pre_tokenizer]
                        )
                elif self.config.pre_tokenizer[-3:] == "SLG":
                    self.tokenizer.pre_tokenizer = eval(f"pre_tokenizers.PreTokenizer.custom({self.config.pre_tokenizer}(semiotic))")
                else:
                    self.tokenizer.pre_tokenizer = eval(f"pre_tokenizers.{self.config.pre_tokenizer}()")
            
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

            model_max_length = self.config.model_max_length,
            model_input_names = ["input_ids","token_type_ids", "attention_mask"],

            unk_token= self.config_vocab.unk_token,
            pad_token= self.config_vocab.pad_token,
            cls_token= self.config_vocab.cls_token,
            sep_token= self.config_vocab.sep_token,
            mask_token= self.config_vocab.mask_token,
        )

        # The following manual addition is due to a bug in HF PreTrainedTokenizerFast, not taking into account the special tokens in the previous declaration
        self.special_tokens = {
            "unk_token": self.config_vocab.unk_token,
            "pad_token": self.config_vocab.pad_token,
            "cls_token": self.config_vocab.cls_token,
            "sep_token": self.config_vocab.sep_token,
            "mask_token": self.config_vocab.mask_token,
            }
        self.bert_tokenizer.add_special_tokens(self.special_tokens)

        # # The following is a test for debugging. I construct a SLG tokenizer to produce a tokenized string separated by spaces to feed to the main tokenized defined as a simple whitespace splitting

        # self.debug_tokenizer = Tokenizer(
        #             models.WordLevel(
        #                 vocab = semiotic.vocab.encode,
        #                 unk_token = self.config_vocab.unk_token
        #                 )
        #                 )
        # self.debug_tokenizer.normalizer = NormalizeSLG

        # self.debug_tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(SequenceSLG(semiotic))

        # self.debug_bert_tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_object = self.debug_tokenizer,
        #     # tokenizer_file=str(semiotic.vocab.path.joinpath("tokenizer.json")), # You can load from the tokenizer file, alternatively

        #     model_max_length = self.config.model_max_length,
        #     model_input_names = ["input_ids","token_type_ids", "attention_mask"],

        #     unk_token= self.config_vocab.unk_token,
        #     pad_token= self.config_vocab.pad_token,
        #     cls_token= self.config_vocab.cls_token,
        #     sep_token= self.config_vocab.sep_token,
        #     mask_token= self.config_vocab.mask_token,
        # )
        # self.debug_bert_tokenizer.add_special_tokens(self.special_tokens)

        # # END OF THE DEBUGGING (but see below)


    def save_tokenizer(self):
        self.tokenizer.save(self.tokenizer_path)

    def build(
        self,
        dataset = None,
        save_tokenized = None,
        n_sents = None,
        return_tokenized = False,
        ):

        # TODO: Add save_tokenized and n_sents to config!!!

        if dataset == None:
            dataset = self.dataset

        if save_tokenized == None:
            save_tokenized = self.config.save_tokenized

        if n_sents == None:
            n_sents = self.config.n_sents

        def tokenize_function(syntagmas):

            # DEBUGGING: pre segment string into a string separated by spaces and then feed it to a whitespace tokenizer

            syntagmas = {"text": [" ".join(self.debug_bert_tokenizer.tokenize(sent)) for sent in syntagmas["text"]]}
            
            # END OF DEBUGGING

            return self.bert_tokenizer(syntagmas["text"], return_special_tokens_mask = True)

        print("SLG: Tokenizing dataset...")

        if n_sents !=None:
            dataset = datasets.DatasetDict({
                "train":dataset["train"].select(range(n_sents)),
                "dev": dataset["dev"].select(range(int(n_sents*(1/self.split_rate[0])*self.split_rate[1]))),
                "test": dataset["test"].select(range(int(n_sents*(1/self.split_rate[0])*self.split_rate[2])))
                })


        tokenized_datasets = dataset.map(
            tokenize_function,
            batched = self.config.input_tokenize["batched"],
            batch_size = self.config.input_tokenize["batch_size"],

            # Using more than 1 proc in iMac with m1 blocks the process
            num_proc = 1, #self.cpu_count, 
            remove_columns = self.config.input_tokenize["remove_columns"]
        )
        if save_tokenized:
            tokenized_datasets.save_to_disk(self.path / "tokenized")
            print(f"SLG: Tokenized corpus saved to {self.path / 'tokenized'}")
        
        print("SLG: Corpus tokenized!")

        if return_tokenized:
            return tokenized_datasets

