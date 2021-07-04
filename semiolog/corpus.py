from datasets import load_dataset
import sklearn
from os.path import isfile

from .syntagmatic import tokenizer
from .util_g import list2txt


class Corpus:
    """
    Corpus class. It takes as argument a Huggingface's dataset object
    """
    
    def __init__(
        self,
        semiotic,
        ) -> None:
        
            self.name = semiotic.name
            self.path = semiotic.paths.corpus
            self.config = semiotic.config.corpus
        
            self.train = None
            self.test = None
            self.dev = None
            
            self.train_len = None
            self.dev_len = None
            self.test_len = None
        

    def from_file(self, path = None):

        if path == None:
            path = self.path
        
        splits = ["train","dev","test"]
        filenames = [(path / f"{fn}.txt") for fn in splits]
        
        for filename in filenames:
            if not isfile(filename):
                return print(f"Warning: {filename} does not exist.\nCorpus will not be loaded from file.\n")

        dataset = load_dataset('text', data_files={
            k:str(fn) for k,fn in zip(splits,filenames)
            })
        
        self.train = dataset["train"]["text"]
        self.test = dataset["test"]["text"]
        self.dev = dataset["dev"]["text"]

        self.train_len = len(self.train)
        self.dev_len = len(self.dev)
        self.test_len = len(self.test)

    def load_dataset(self,dataset = None):
        if dataset == None:
            dataset = self.config.dataset
        if isinstance(dataset,list):
            data = load_dataset(*dataset)
        else:
            data = load_dataset(dataset)
        return data
    
    def build(
        self,
        dataset_name = None,
        length = None,
        save = False,
        ):
        
        if dataset_name == None:
            if self.config.dataset == None:
                return print("Error: No dataset defined")
            dataset_name = self.config.dataset
        
        if length == None:
            length = self.config.length

        dataset = self.load_dataset(dataset_name)
        
        if "test" in dataset:
            self.train = dataset["train"][:length]["text"] # TODO: split into sentences
            
            if "dev" or "validation" in dataset:
                self.test = dataset["test"][:length]["text"] # TODO: split into sentences
                self.dev = (dataset.get("dev",dataset["validation"]))[:length]["text"] # TODO: split into sentences
            else:
                dataset["test"]
        
        else:
            input = dataset["train"][:length]["text"]
            
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
        
        print("Corpus built")
        if save == True:
            self.save()
            print(f"Corpus saved to {self.path}")
        

        
    def save(self, path = None):
        
        if path == None:
            path = self.path

        list2txt(self.train,"train", path)
        list2txt(self.dev,"dev", path)
        list2txt(self.test,"test", path)
    

        