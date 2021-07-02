from datasets import load_dataset
import sklearn

from . import paths
from .syntagmatic import tokenizer
from .util_g import list2txt


class Corpus:
    """
    Corpus class. It takes as argument a Huggingface's dataset object
    """
    
    def __init__(
        self,
        dataset,
        length = None,
        saved_corpus = False,
        ) -> None:
        
        if saved_corpus:
            self.train = dataset["train"]["text"]
            self.test = dataset["test"]["text"]
            self.dev = dataset["dev"]["text"]
            
        else:    
            self.dataset = dataset
            if "test" in self.dataset:
                self.train = self.dataset["train"] # TODO: split into sentences
                
                if "dev" or "validation" in self.dataset:
                    self.test = self.dataset["test"] # TODO: split into sentences
                    self.dev = self.dataset.get("dev",self.dataset["validation"]) # TODO: split into sentences
                else:
                    self.dataset["test"]
            
            else:
                input = " ".join(self.dataset["train"][:length]["text"])
                
                normal = tokenizer.normalizers.NFKC.normalize(None,input)
                pre_token = tokenizer.pre_tokenizers.Layout.pre_tokenize(None,normal)
                process = tokenizer.processors.SentencesNLTK.process(None,pre_token,is_pretokenized=True)
                
                self.sentences = tokenizer.post_processors.WikiFR.post_process(None,process)
                self.train, dev_test = sklearn.model_selection.train_test_split(list(self.sentences), train_size=.9, test_size=.1)
                self.dev, self.test = sklearn.model_selection.train_test_split(dev_test, train_size=.5, test_size=.5)
        
        self.train_len = len(self.train)
        self.dev_len = len(self.dev)
        self.test_len = len(self.test)
        
    def save(self,corpus_name):
        list2txt(self.train,"train", paths.corpora / corpus_name)
        list2txt(self.dev,"dev", paths.corpora / corpus_name)
        list2txt(self.test,"test", paths.corpora / corpus_name)
            
            

def load_corpus(corpus_name: str,path=paths.corpora):
    
    dataset = load_dataset('text', data_files={
        'train': str(path / corpus_name / "train.txt"),
        'test': str(path / corpus_name / "test.txt"),
        'dev': str(path / corpus_name / "dev.txt"),
        })
    
    return Corpus(dataset,length = None, saved_corpus = True)
    
            
            
            
        
        # normalizer = tokenizer.normalizers.Sequence(["Lowercase","StripPunctuation","StripWhitespaces"])
        # self.train_norm = normalizer.normalize("".join(self.train))
        
        