import inspect
import sys

from .util_g import dict2json, json2dict

# TODO: Solve version as global variable
slg_version = "0.1"

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


class General(Section):
    
    def __init__(self, semiotic) -> None:
        self.name = semiotic.name
        # self.paths = Paths(semiotic.name)
        self.dataset = None


class Vocabulary(Section):
    
    def __init__(self, semiotic) -> None:
        self.vocFileName = None
        self.nGramFileName = None
        self.specialTokens = None

        
class Syntagmatic(Section):
    """
    Possible Normalizers: "NFKD","Lowercase","StripPunctuation","StripWhitespaces"
    
    Possible Processors: "SequenceSLG" "TreeSLG", "StripWhitespaces"m
    """
    def __init__(self, semiotic) -> None:
        self.normalizer = None
        self.pre_tokenizer = None
        self.processor = None
        self.post_processor = None


class Paradigmatic(Section):
    def __init__(self, semiotic) -> None:
        self.model = None
        self.top_k = None
        self.exclude_punctuation = None
        self.cumulative_sum_threshold = None


class Evaluation(Section):
    def __init__(self, semiotic) -> None:
        self.ud_model = None
        self.cp_model = None