from .util_g import dict2json
from .paths import Paths

class Config:
    
    def __init__(self, semiotic) -> None:
        self.general = General(semiotic.name)
        self.vocabulary = Vocabulary()
        self.chain = Chain()
        self.paradigm = Paradigm()
        self.evaluation = Evaluation()

        self.paths = Paths(self.general.name)


    # def __repr__(self) -> str:
    #     return str(self.__dict__)

    def save(self):
        """
        Saves the configuration asa JSON file in the root of the current model. Paths are not saved
        """
        config_dict = {k: v.__dict__ for k,v in self.__dict__.items()if k != "paths"}

        dict2json(config_dict,"config", self.paths.semiotic)

class General:
    def __init__(self,name) -> None:
        self.name = name
        # self.paths = Paths(name)
        self.testSentences = None
    
    # def __repr__(self) -> str:
    #     return str(self.__dict__)

class Vocabulary:
    def __init__(self) -> None:
        self.vocFileName = None

    # def __repr__(self) -> str:
    #     return str(self.__dict__)

class Chain:
    def __init__(self) -> None:
        self.segType = None
    
    # def __repr__(self) -> str:
    #     return str(self.__dict__)

class Paradigm:
    def __init__(self) -> None:
        self.model = None
        self.top_k = None
    
    # def __repr__(self) -> str:
    #     return str(self.__dict__)

class Evaluation:
    def __init__(self) -> None:
        self.ud_model = None
        self.cp_model = None
    
    # def __repr__(self) -> str:
    #     return str(self.__dict__)