from thinc.api import Config

from datasets import load_dataset
import sklearn

from . import paths
from .syntagmatic import tokenizer


class Corpus:
    """
    Corpus class. It uses huggingface's dataset library
    """
    
    def __init__(self,path,name,split = None,model_name=None,length = None) -> None:
        # self.config = Config().from_disk(paths.corpora / model_name / "config.cfg")
        
        self.dataset = dict(load_dataset(
            path,
            name,
            split = split
        ))
        if "test" in self.dataset:
            self.train = self.dataset["train"] # TODO: split into sentences
            
            if "dev" or "validation" in self.dataset:
                self.test = self.dataset["test"] # TODO: split into sentences
                self.dev = self.dataset.get("dev",self.dataset["validation"]) # TODO: split into sentences
            else:
                self.dataset["test"]
        
        else:
            input = " ".join(self.dataset["train"]["text"][:length])
            pre_token = tokenizer.pre_tokenizers.Layout.pre_tokenize(None,input)
            process = tokenizer.processors.SentencesNLTK.process(None,pre_token,is_pretokenized=True)
            self.sentences = tokenizer.post_processors.WikiFR.post_process(None,process)
            self.train, dev_test = sklearn.model_selection.train_test_split(list(self.sentences), train_size=.9, test_size=.1)
            self.dev, self.test = sklearn.model_selection.train_test_split(dev_test, train_size=.5, test_size=.5)
        
        normalizer = tokenizer.normalizers.Sequence(["NFKD","Lowercase","StripPunctuation","StripWhitespaces"])
        self.train_norm = normalizer.normalize("".join(self.train))
        
        