from datasets import load_dataset
import sklearn

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
                self.train = self.dataset["train"][:length]["text"] # TODO: split into sentences
                
                if "dev" or "validation" in self.dataset:
                    self.test = self.dataset["test"][:length]["text"] # TODO: split into sentences
                    self.dev = (self.dataset.get("dev",self.dataset["validation"]))[:length]["text"] # TODO: split into sentences
                else:
                    self.dataset["test"]
            
            else:
                input = self.dataset["train"][:length]["text"]
                
                # TODO: the following could be parallelized but so far no efficiency gain
                sentences = []
                for document in input:
                    normal = tokenizer.normalizers.NFKC.normalize(None,document)
                    pre_token = tokenizer.pre_tokenizers.Layout.pre_tokenize(None,normal)
                    process = tokenizer.processors.SentencesNLTK.process(None,pre_token,is_pretokenized=True)
                    doc_sentences = tokenizer.post_processors.WikiFR.post_process(None,process)
                    sentences.extend(doc_sentences)
                
                self.sentences = sentences
                self.train, dev_test = sklearn.model_selection.train_test_split(list(self.sentences), train_size=.9, test_size=.1)
                self.dev, self.test = sklearn.model_selection.train_test_split(dev_test, train_size=.5, test_size=.5)
        
        self.train_len = len(self.train)
        self.dev_len = len(self.dev)
        self.test_len = len(self.test)
        
    def save(self,corpus_name, path):
        list2txt(self.train,"train", path)
        list2txt(self.dev,"dev", path)
        list2txt(self.test,"test", path)

            

def load_corpus(corpus_name: str,path):
    
    dataset = load_dataset('text', data_files={
        'train': str(path / "train.txt"),
        'test': str(path / "test.txt"),
        'dev': str(path / "dev.txt"),
        })
    
    return Corpus(dataset,length = None, saved_corpus = True)
        