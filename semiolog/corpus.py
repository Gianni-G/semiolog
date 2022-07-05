from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from datasets.utils import disable_progress_bar
# disable_progress_bar()

from os.path import isfile, isdir, basename
from os import listdir
from nltk.tokenize import sent_tokenize

import socket
socket_name = socket.gethostname()
if any(name in socket_name for name in {"Gianni","vpn"}):
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange

from .util import load_file, save_file, flatten

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

            self.dataset = DatasetDict()
        

    def from_file(self, path = None, test_only = False):

        # TODO: loading only part of the dataset (corresponding to config.corpus.length) is not yet implemented

        if path == None:
            path = self.path
        
        if isdir(path / "dataset"):
            self.dataset = load_from_disk(path / "dataset")
            print(f"SLG [I]: Dataset loaded from disk (dataset file)")
            if not test_only:
                self.train = self.dataset["train"]
                self.train_len = self.train.num_rows

        else:
            splits = ["train","dev","test"] if test_only == False else ["dev","test"]
            filenames = [(path / f"{fn}.txt") for fn in splits]
            
            for filename in filenames:
                if not isfile(filename):
                    return print(f"SLG [W]: {filename} does not exist.\nCorpus will not be loaded from disk.\n")

            if not test_only:
                self.dataset = self.load_dataset({"train": "train.txt", "dev": "dev.txt", "test": "test.txt"})
                print(f"SLG [I]: Dataset loaded from disk (TXT files)")
                

                self.train = self.dataset["train"]
                self.train_len = self.train.num_rows

            else:
                self.dataset = self.load_dataset({"dev": "dev.txt", "test": "test.txt"})
        
        self.dev = self.dataset["dev"]
        self.test = self.dataset["test"]

        self.dev_len = self.dev.num_rows
        self.test_len = self.test.num_rows

    def load_dataset(
        self,
        dataset = None,
        original=False,
        keep_source=False
        ):

        if original:
            load_path = self.path / 'original'
        else:
            load_path = self.path

        if dataset == None:
            dataset = [fn for fn in listdir(load_path) if fn.endswith(".txt")]


        if keep_source:
            if not original:
                print("SLG [W]: 'keep_source' is True for non original corpus files. Dataset will be loaded without a source feature. If you want a source feature in your dataset, please make sure to load orignal files.")
            else:
                text = []
                source = []
                for fn in dataset:
                    # treatise_raw = txt2list(fn[:-4],self.path / 'original')
                    treatise_raw = load_file(self.path / f"original/{fn}")
                    text += treatise_raw
                    source += [fn[:-4]]*len(treatise_raw)
                
                data = DatasetDict({"train": Dataset.from_dict({"text":text, "source":source})})
        else:

            if len(dataset) == 1:
                dataset = dataset[0]

            data = load_dataset(str(load_path), data_files=dataset)

        return data


    def build(
        self,
        dataset = None,
        length = None,
        save = False,
        split_rate = None,
        split_sent = False,
        keep_source = False,
        ):
        
        if dataset == None:
            dataset = self.config.dataset

        if dataset == None:
            dataset = [fn for fn in listdir(self.path / 'original') if fn.endswith(".txt")]
            if len(dataset) == 1:
                dataset = dataset[0]

        if length == None:
            length = self.config.length
            
        if split_rate == None:
            split_rate = self.config.split_rate

        assert self.config.dataset != None or dataset != [], f"SLG [E]: No dataset defined or no txt files found in the model's folder."
        

        # if keep_source:
        #     text = []
        #     source = []
        #     for fn in dataset:
        #         treatise_raw = txt2list(fn[:-4],self.path / 'original')
        #         if split_sent:
        #             treatise_sents = []
        #             for paragraph in treatise_raw:
        #                 treatise_sents += sent_tokenize(paragraph)
        #         else:
        #             treatise_sents = treatise_raw
        #         text += treatise_sents
        #         source += [fn[:-4]]*len(treatise_sents)


        #     self.dataset = Dataset.from_dict({"text":text, "source":source})

        # else:
        if len(dataset) == 1:
            dataset = dataset[0]

        self.dataset = self.load_dataset(dataset, original=True, keep_source=keep_source)
        print(f"\nSLG [I]: Dataset loaded from the following files: {dataset}.\n")
        
        # TODO: The splitting should be generalized to handle any kind of feature potentially existing in a database
        if split_sent:
            print("Spliting original text into sentences.")

            if keep_source:
                def split(batch):
                    text_src = [(sent_tokenize(sent),src) for sent,src in zip(batch["text"],batch["source"])]
                    text = flatten([t for t,s in text_src])
                    source = flatten([[s]*len(t) for t,s in text_src])
                    split_dict = {"text": text, "source": source}
                    return split_dict
                    
                for column in self.dataset.column_names:
                    self.dataset[f"{column}"] = self.dataset[f"{column}"].map(split, remove_columns=["text","source"], batched=True)

            else:
                for column in self.dataset.column_names:
                    self.dataset[f"{column}"] = self.dataset[f"{column}"].map(lambda batch: {"text": flatten([sent_tokenize(sent) for sent in batch["text"]])}, remove_columns=["text"], batched=True)

        if "test" in self.dataset:
            if length != None:
                split_lengths = tuple([int(length*r) for r in split_rate])
            print("This feature has not been tested yet. Please check")

            #TODO: Check the entire "if". Use "train_test_split" instead 

            self.train = self.dataset["train"][:split_lengths[0]]
            
            if "dev" or "validation" in self.dataset:
                
                self.dev = self.dataset.get("dev",self.dataset["validation"])[:split_lengths[1]]
                self.test = self.dataset["test"][:split_lengths[2]]
                
            else:
                
                split_test = self.dataset["test"][:split_lengths[1]+split_lengths[2]].train_test_split(split_rate[0]*split_rate[1])

                self.dev = split_test["train"]
                self.test = split_test["test"]
            
            self.dataset = DatasetDict({"train": self.train, "dev": self.dev, "test": self.test})
                    
        
        else:

            dataset_train = self.dataset["train"].train_test_split(test_size=sum(split_rate[1:]))

            dataset_test = dataset_train["test"].train_test_split(split_rate[1]/sum(split_rate[1:]))

            self.dataset = DatasetDict({"train": dataset_train["train"], "dev": dataset_test["train"], "test": dataset_test["test"]})
            
            self.sentences = self.dataset["train"]["text"]

            self.train = self.dataset["train"]
            self.dev = self.dataset["dev"]
            self.test = self.dataset["test"]
            
        self.train_len = self.train.num_rows
        self.dev_len = self.dev.num_rows
        self.test_len = self.test.num_rows
        
        print("Corpus built")
        if save:
            if keep_source:
                self.save(dataset_format=True)
                print(f"Corpus saved as dataset file to {self.path / 'dataset'}")

            else:
                self.save()
                print(f"Corpus saved as TXT files to {self.path}")
        
        
    def save(self, path = None, dataset_format=False):
        
        if path == None:
            path = self.path

        if dataset_format:
            self.dataset.save_to_disk(path / "dataset")
        else:
            save_file(self.train["text"], path / "train.txt")
            save_file(self.dev["text"], path / "dev.txt")
            save_file(self.test["text"], path / "test.txt")