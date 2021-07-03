from .util_g import dict2json, json2dict
from .paths import Paths

class Config:
    
    def __init__(self, semiotic) -> None:
        self.general = General(semiotic.name)
        self.vocabulary = Vocabulary()
        self.syntagmatic = Syntagmatic()
        self.paradigmatic = Paradigmatic()
        self.evaluation = Evaluation()

        self.paths = Paths(self.general.name)


    def __repr__(self) -> str:
        return str(self.__dict__)

    def save(self):
        """
        Saves the configuration asa JSON file in the root of the current model. Paths are not saved
        """
        config_dict = {k: v.__dict__ for k,v in self.__dict__.items()if k != "paths"}

        dict2json(config_dict,"config", self.paths.semiotic)
    
    def from_file(self,paths):
        config_dict = json2dict("config",paths)
        
        for section in config_dict:
            print(section)
            for key in config_dict[section]:
                print(f"{key}: {config_dict[section][key]}")
                setattr(eval(f"self.{section}"), key, config_dict[section][key])

class General:
    def __init__(self,name) -> None:
        self.name = name
        # self.paths = Paths(name)
        self.testSentences = None
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class Vocabulary:
    def __init__(self) -> None:
        self.vocFileName = None
        self.nGramFileName = None
        self.specialTokens = None
    
    def __repr__(self) -> str:
        return str(self.__dict__)
        
class Syntagmatic:
    """
    Possible Normalizers: "NFKD","Lowercase","StripPunctuation","StripWhitespaces"
    
    Possible Processors: "SequenceSLG" "TreeSLG", "StripWhitespaces"m
    """
    def __init__(self) -> None:
        self.normalizer = None
        self.pre_tokenizer = None
        self.processor = None
        self.post_processor = None

    def __repr__(self) -> str:
        return str(self.__dict__)

class Paradigmatic:
    def __init__(self) -> None:
        self.model = None
        self.top_k = None
        self.exclude_punctuation = None
        self.cumulative_sum_threshold = None
    
    def __repr__(self) -> str:
        return str(self.__dict__)

class Evaluation:
    def __init__(self) -> None:
        self.ud_model = None
        self.cp_model = None
    
    def __repr__(self) -> str:
        return str(self.__dict__)