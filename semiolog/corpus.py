from datasets import load_dataset, DatasetDict
from os.path import isfile
from os import listdir
    
from .util import list2txt


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
        
        splits = ["train","dev","test"] if test_only == False else ["dev","test"]
        filenames = [(path / f"{fn}.txt") for fn in splits]
        
        for filename in filenames:
            if not isfile(filename):
                return print(f"SLG Warning: {filename} does not exist.\nCorpus will not be loaded from file.\n")

        if not test_only:
            self.dataset = self.load_dataset({"train": "train.txt", "dev": "dev.txt", "test": "test.txt"})

            self.train = self.dataset["train"]
            self.train_len = self.train.num_rows

        else:
            self.dataset = self.load_dataset({"dev": "dev.txt", "test": "test.txt"})
        
        self.dev = self.dataset["dev"]
        self.test = self.dataset["test"]

        self.dev_len = self.dev.num_rows
        self.test_len = self.test.num_rows

    def load_dataset(self, dataset = None, original=False):

        if original:
            load_path = self.path / 'original'
        else:
            load_path = self.path

        if dataset == None:
            dataset = [fn for fn in listdir(load_path) if fn.endswith(".txt")]
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

        if self.config.dataset == None and dataset == []:
            return print("SLG: Error: No dataset defined or no txt files found in the model's folder.")
        
        self.dataset = self.load_dataset(dataset, original=True)
        print(f"\nSLG: Dataset loaded from the follwing files: {dataset}.\n")
        
        if "test" in self.dataset:
            if length != None:
                split_lengths = tuple([int(length*r) for r in split_rate])
            print("This feature has not been tested yet. Pleas check")

            #TODO: Check the entire "if"

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

            dataset_train = self.dataset["train"].train_test_split(sum(split_rate[1:]))

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
        if save == True:
            self.save()
            print(f"Corpus saved to {self.path}")
        
        
    def save(self, path = None):
        
        if path == None:
            path = self.path

        list2txt(self.train["text"],"train", path)
        list2txt(self.dev["text"],"dev", path)
        list2txt(self.test["text"],"test", path)