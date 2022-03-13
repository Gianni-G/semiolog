import inspect
import sys
from os.path import isfile

from .util import dict2json, json2dict

# TODO: Solve version as global variable
slg_version = "0.2"

class Config:
    
    def __init__(self, semiotic) -> None:
        """
        Automatically loads all classes below as section attributes of the class Config
        """
        
        self.sections = [s for s,repr in inspect.getmembers(sys.modules[__name__], inspect.isclass) if s not in ("Config","Section")]
        
        for section in self.sections:
            exec(
                f"self.{section.lower()} = {section}(semiotic)"
            )

        self.path = semiotic.paths.semiotic

    def __repr__(self) -> str:
        return str(self.__dict__)

    def save(self):
        """
        Saves the configuration asa JSON file in the root of the current model. Paths are not saved
        """
        config_dict = {k: v.__dict__ for k,v in self.__dict__.items() if k in {s.lower() for s in self.sections}}

        dict2json(config_dict,"config", self.path)
    
    def from_file(self,path = None):
        if path == None:
            path = self.path
        
        filename = str(path / "config.json")
        if not isfile(filename):
            return print(f"SLG Warning: {filename} does not exist.\Config will not be loaded from file.\n")
            
        config_dict = json2dict("config",path)
        
        for section in config_dict:
            for key in config_dict[section]:
                setattr(eval(f"self.{section}"), key, config_dict[section][key])


class Section:
    
    def __init__(self, semiotic) -> None:
        pass
    
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    
class System(Section):
    
    def __init__(self, semiotic) -> None:
        self.slg_version = slg_version
        self.cpu_count = None
        self.multiple_gpus = False


class General(Section):
    """
    Possible tensor implementations: 'tf' (TensorFlow), 'pt' (PyTorch), 'np' (numpy)
    """
    def __init__(self, semiotic) -> None:
        self.name = semiotic.name
        self.dataset = None
        self.tensor_implementation = "tf"

class Corpus(Section):
    
    def __init__(self, semiotic) -> None:
        self.dataset = None
        self.split_rate = (.9,.05,.05)
        self.length = None
        self.normalizer = None
        self.pre_tokenizer = None
        self.processor = None
        self.post_processor = None
        
class Vocabulary(Section):
    
    def __init__(self, semiotic) -> None:
        """
        By default, the special tokens are:
        [PAD], [UNK], [CLS], [SEP], [MASK]
        """
        self.size = None
        self.normalizer = None
        self.special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
            ]
        self.unk_token="[UNK]"
        self.pad_token="[PAD]"
        self.cls_token="[CLS]"
        self.sep_token="[SEP]"
        self.mask_token="[MASK]"



        
class Syntagmatic(Section):
    """
    Possible Normalizers: "NFKD","Lowercase","StripPunctuation","StripWhitespaces"
    
    Possible Processors: "SequenceSLG" "TreeSLG", "StripWhitespaces"m
    """
    def __init__(self, semiotic) -> None:
        self.from_file = False
        self.normalizer = None
        self.pre_tokenizer = None
        self.processor = "WordLevel"
        self.post_processor = None
        self.model_max_length = None


class Paradigmatic(Section):
    """
    'learning_rate' and 'weight_decay'
    """
    def __init__(self, semiotic) -> None:

        self.load_pretrained = False

        self.huggingface_pretrained = None

        self.model_config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "use_cache": True,
            "classifier_dropout": None,

            "learning_rate": 2e-5,
            "weight_decay": 0.01
        }
        
        self.optimizer = {
            "optimizer": "AdamWeightDecay",

            "learning_rate": 2e-5,
            "weight_decay": 0.01
        }

        self.mask_probability = 0.15

        self.input_tokenize = {
            "batched": True,
            "batch_size": 512,
            "remove_columns": ["text"],
        }

        self.input_sets = {
            "shuffle": True,
            "batch_size": 16
        }

        self.training_epochs = 2

        self.save = True

        self.model = None
        self.top_k = None
        self.exclude_punctuation = None
        self.cumulative_sum_threshold = None


class Evaluation(Section):
    def __init__(self, semiotic) -> None:
        self.ud_model = None
        self.cp_model = None