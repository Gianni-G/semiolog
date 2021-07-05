from datasets import load_dataset
import sklearn
from os.path import isfile
from tqdm.auto import tqdm

from .syntagmatic import tokenizer
from .util import list2txt, txt2list


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
        

    def from_file(self, path = None, test_only = False):

        if path == None:
            path = self.path
        
        splits = ["train","dev","test"] if test_only == False else ["dev","test"]
        filenames = [(path / f"{fn}.txt") for fn in splits]
        
        for filename in filenames:
            if not isfile(filename):
                return print(f"Warning: {filename} does not exist.\nCorpus will not be loaded from file.\n")

        self.test = txt2list("test", self.path)
        self.dev = txt2list("dev", self.path)

        self.dev_len = len(self.dev)
        self.test_len = len(self.test)

        if not test_only:
            self.train = txt2list("train", self.path)
            self.train_len = len(self.train)


    def load_dataset(self,dataset = None):
        if dataset == None:
            dataset = self.config.dataset
        if isinstance(dataset,list):
            data = load_dataset(*dataset)
        else:
            data = load_dataset(dataset)
        return data

    def pre_process_document(self, document):
        normal = tokenizer.normalizers.NFKC.normalize(None,document)
        pre_token = tokenizer.pre_tokenizers.Layout.pre_tokenize(None,normal)
        process = tokenizer.processors.SentencesNLTK.process(None,pre_token,is_pretokenized=True)
        doc_sentences = tokenizer.post_processors.WikiFR.post_process(None,process)
        return doc_sentences
    
    
    # TODO: the following could be parallelized but so far no efficiency gain
    
    def pre_process_corpus(self, input, progress_bar = True):
        sentences = []
        for document in tqdm(input, disable = not progress_bar):
            doc_sentences = self.pre_process_document(document)
            sentences.extend(doc_sentences)
        return sentences   


    def build(
        self,
        dataset = None,
        length = None,
        save = False,
        progress_bar = True,
        split_rate = None,
        ):
        
        if dataset == None:
            dataset = self.config.dataset
        
        if length == None:
            length = self.config.length
            
        if split_rate == None:
            split_rate = self.config.split_rate


        if self.config.dataset == None:
            return print("Error: No dataset defined")
        
        dataset_ = self.load_dataset(dataset)
        
        if "test" in dataset_:
            if length != None:
                split_lengths = tuple([int(length*r) for r in split_rate])
            print("This feature has not been tested yet. Pleas check")
            #Check the entire "if"
            
            input = dataset_["train"][:split_lengths[0]]["text"]
            self.train = self.pre_process_corpus(input, progress_bar = progress_bar)
            
            if "dev" or "validation" in dataset_:
                
                input = (dataset_.get("dev",dataset_["validation"]))[:split_lengths[1]]["text"]
                
                self.dev = self.pre_process_corpus(input, progress_bar = progress_bar)
                
                input = dataset_["test"][:split_lengths[2]]["text"] 
                
                self.test = self.pre_process_corpus(input, progress_bar = progress_bar)
                
            else:
                input = dataset_["test"][:split_lengths[1]+split_lengths[2]]["text"]
                
                self.dev, self.test = sklearn.model_selection.train_test_split(
                    
                    self.pre_process_corpus(input, progress_bar = progress_bar),
                    
                    train_size=split_rate[0]*split_rate[1], test_size=split_rate[0]*split_rate[2])
                    
        
        else:
            input = dataset_["train"][:length]["text"]
            
            self.sentences = self.pre_process_corpus(input, progress_bar = progress_bar)
            
            self.train, dev_test = sklearn.model_selection.train_test_split(list(self.sentences), train_size=split_rate[0], test_size=split_rate[1]+split_rate[2])
            
            self.dev, self.test = sklearn.model_selection.train_test_split(dev_test, train_size=split_rate[0]*split_rate[1], test_size=split_rate[0]*split_rate[2])
            
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
    

        