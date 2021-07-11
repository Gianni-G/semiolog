from collections import Counter
import csv
from typing import Union, Iterable, Dict, Any
from tqdm.notebook import trange, tqdm #tqdm.auto
import regex as re
from os import makedirs
from os.path import isfile, isdir
from functools import reduce
import operator
from multiprocessing import cpu_count, Pool

# import psutil

from . import util
from .syntagmatic import tokenizer

# TODO: Solve version as global variable
slg_version = "0.1"

class Vocabulary:
    
    def __init__(self,semiotic):
        
        #TODO: Is there another way than loading the corpus (or the semiotic) here?
        self.corpus = semiotic.corpus
        
        self.name = semiotic.name
        self.path = semiotic.paths.vocabulary
        self.config = semiotic.config.vocabulary
        self.cpu_count = semiotic.config.system.cpu_count
        
        self.merges = None
        self.encode = None
        self.freq = None
        self.alpha = None
        
        self.decode = None
        
        self.len = None
        self.freq_mass = None
        self.prob = None


    def from_file(self,path = None):
        if path == None:
            path = self.path


        filenames = [(path / fn) for fn in ["merges.txt","vocab.json","freq.json","alpha.json"]]
        
        for filename in filenames:
            if not isfile(filename):
                return print(f"Warning: {filename} does not exist.\nVocabulary will not be loaded from file.\n")
        
        self.merges = util.txt2list("merges",path)[1:] # The first line needs to be stripped
        self.encode = util.json2dict("vocab",path)
        self.freq = util.json2dict("freq",path)
        self.alpha = util.json2dict("alpha",path)

        self.decode = {i:k for k,i in self.encode.items()}
        
        self.len = len(self.encode)
        self.freq_mass = sum(self.freq.values())
        self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}


    def __repr__(self) -> str:
        return f"Voc({self.freq})"

    def __str__(self) -> str:
        return str(self.freq)

    def __getitem__(self, item):
         return self.prob[item]

    def head(self,size=10):
        return list(self.freq.items())[:size]
    
    def tail(self,size=10):
        return list(self.freq.items())[-size:]

    def alphabetic(self):
        pass

    def keys(self):
        pass

    def values(self):
        pass
    
    def build(
        self,
        corpus = None,
        vocab_size = None,
        special_tokens = None,
        save = False,
        save_step = None,
        progress_bar = True,
        resume_merges = False,
        parallel = False,
        parallel_mode = "process"
        ):
        """
        Build vocabulary from a Corpus.
        Vocabularies can be extended by providing an existing merging list. If resume_merges = True, the current merges in self.merges will be used. Otherwise one can provide a list of merges as value of resume_merges.
        """
        
        if corpus == None:
            corpus = self.name
        
        if vocab_size == None:
            vocab_size = self.config.size
        
        if special_tokens == None:
            special_tokens = self.config.special_tokens
        
        if save == True and save_step != None:
            saveQ = True
            
            if not isdir(self.path):
                makedirs(self.path)
                
            save_steps = {save_step*i for i in range(int(vocab_size/save_step)+1)}
        else:
            saveQ = False
        
        
        #TODO: find_best_pair must be parallelizable, but no gain of efficiency so far
        def find_best_pair_original(
            chain_spaced,
            parallel = False,
            parallel_mode = "process"):
            
            pre_units = chain_spaced.split()
            pre_units_pairs = zip(pre_units, pre_units[1:])
            
            if parallel == False:
                pairs = Counter(pre_units_pairs)
                
                return pairs.most_common()[0]
            else:
                
                chunk_size = util.chunk_size(len(pre_units),self.cpu_count)
                pairs_chunks = util.chunks(pre_units_pairs, chunk_size)
                
                if parallel_mode == "thread":

                    # result = util.multithreading(Counter,pre_units_pairs,chunk_size)
                    
                    result = util.multithreading(Counter,pairs_chunks)
                    
                else:
                                        
                    # result = util.multiprocessing(Counter,pre_units_pairs,chunk_size) 
                    
                    result = util.multiprocessing(Counter,pairs_chunks) 
                                        
                pairs = reduce(operator.add, result)
            
            return pairs.most_common()[0]
        
        def find_best_pair(
            chain_list,
            parallel = False,
            parallel_mode = "process"):
            
            if parallel:
                chunk_size = util.chunk_size(len(chain_list),self.cpu_count)
                
            # chain_list = zip(chain_list, chain_list[1:])
            
            if parallel == False:
                pair_count = Counter()
                for pair in tqdm(list(zip(chain_list, chain_list[1:])),desc="Find Best Pair:", leave = False,disable= not progress_bar):
                # for pair in list(zip(chain_list, chain_list[1:])):
                    pair_count[pair] += 1
                
                return pair_count.most_common()[0]
            
            #TODO: rewrite parallelize to reflect changes from chain to chain list
            else:
                
                chain_list = util.chunks(chain_list, chunk_size)
                
                if parallel_mode == "thread":

                    # result = util.multithreading(Counter,pairs,chunk_size)
                    
                    result = util.multithreading(Counter,chain_list)
                    
                else:
                                        
                    # result = util.multiprocessing(Counter,pairs,chunk_size) 
                    
                    result = util.multiprocessing(Counter,chain_list) 
                                        
                chain_list = reduce(operator.add, result)
            
            return chain_list.most_common()[0]
        
        def agglutinate_chain(pair, chain_list):
            chain_list = " ".join(chain_list) 
            bigram = re.escape(" ".join(pair))
            p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            chain_list = p.sub("".join(pair), chain_list)
            chain_list = chain_list.split()
            return chain_list

        def agglutinate_chain_new(pair, chain_list):
            pair = list(pair)
            for i in trange(len(chain_list)):
                if chain_list[i:i+2] == pair:
                    chain_list[i] = "".join(pair)
                    del chain_list[i+1]
            
            return chain_list
        
        if isinstance(self.config.normalizer,list):
            normalizer = eval(
                f"tokenizer.normalizers.Sequence({self.config.normalizer})"
                )
        else:
            normalizer = eval(
                f"tokenizer.normalizers.{util.if_none_disable(self.config.normalizer)}"
            )

        
        # chain = ""
        # for sent in tqdm(self.corpus.train):
        #     sent = normalizer.normalize(None,sent)
        #     sent = " ".join(sent)
        #     if sent !="":
        #         chain += " " + sent
        # # chain = chain.strip()
        
        chain_list = []
        alphabet = Counter()
        for sent in tqdm(self.corpus.train, desc="Chain List & Alphabet:", disable = not progress_bar):
            sent = normalizer.normalize(sent)
            sent = list(sent)
            if sent !=[]:
                chain_list += sent
                alphabet.update(Counter(sent))
        
        
            
        # alphabet = Counter()
        # for char in tqdm(chain):
        #     alphabet[char] += 1
        
        if resume_merges != False:
            if resume_merges == True:
                merges = self.merges
            elif isinstance(resume_merges,list):
                merges = resume_merges
            
            for pair in tqdm(merges, desc = "Resuming Existing Vocabulary",disable = not progress_bar):
                chain_list = agglutinate_chain(tuple(pair.split()),chain_list)
            vocabulary = Counter()
            for term in tqdm(chain_list,desc="Building Resumed Vocabulary", disable = not progress_bar):
                vocabulary[term] += 1
            
        else:
            merges = []
            vocabulary = alphabet
        
        
        special_tokens_len = 0 if special_tokens == None else len(special_tokens)
        voc_len = len(vocabulary) + special_tokens_len
        pair = None
        
        
        t = trange(vocab_size - voc_len, disable = not progress_bar)
        for i in t:
            t.set_description(f"Pair: {pair})\n")
            t.refresh()
            
            pair = find_best_pair(
                chain_list,
                parallel = parallel,
                parallel_mode = parallel_mode
                )
            
            chain_list = agglutinate_chain(pair[0], chain_list)
            
            merges.append(" ".join(pair[0]))
            
            if saveQ == True:
                if voc_len + i + 1 in save_steps:
                    vocabulary = Counter(chain_list).most_common()
                    if special_tokens != None:
                        vocabulary = vocabulary + [(token,0) for token in special_tokens]

                    self.merges = merges
                    self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
                    self.freq = dict(vocabulary)
                    self.alpha = dict(alphabet)
                    step_path = self.path / str(voc_len+i+1)
                    self.save(step_path)
                    print(f"Intermediate vocabulary saved to {step_path}")
        
        vocabulary = Counter()            
        for term in tqdm(chain_list,desc="Building Final Vocabulary", disable = not progress_bar):
            vocabulary[term] += 1
        vocabulary = vocabulary.most_common()
        
        if special_tokens != None:
            vocabulary = vocabulary + [(token,0) for token in special_tokens]

        self.merges = merges
        self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
        self.freq = dict(vocabulary)
        self.alpha = dict(alphabet.most_common())

        self.decode = {i:k for k,i in self.encode.items()}
        
        self.len = len(vocabulary)     
        self.freq_mass = sum(self.freq.values())
        self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

        print("Vocabulary built")
        
        if save == True:
            self.save()
            print(f"Vocabulary saved to {self.path}")
    
    def save(self, path = None):
        
        if path == None:
            path = self.path
            
        version_stamp = f"#version: {slg_version} - Built by `semiolog`"
        
        util.list2txt([version_stamp]+self.merges,"merges", path)
        util.dict2json(self.encode,"vocab", path)
        util.dict2json(self.freq,"freq", path)
        util.dict2json(self.alpha,"alpha", path)
        


class nGram(Vocabulary):
    def __init__(self, filename = None, special_tokens = None):

        if filename != None:
                
            with open(filename, "r") as f:
                csv_reader = csv.reader(f)
                voc = Counter()
                for line in csv_reader:
                    voc[tuple(line[:2])] = int(line[-1])
                voc = dict(voc.most_common())

            self.filename = filename
            self.len = len(voc)
            self.freq = voc
            self.freq_mass = sum(voc.values())
            self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

            self.encode = {k:i for i,(k,v) in enumerate(voc.items())}
            self.decode = {i:k for k,i in self.encode.items()}
        else:
            pass

    def __repr__(self) -> str:
        return f"nGram({self.freq})"

    def __str__(self) -> str:
        return str(self.freq)

    def __getitem__(self, item):
         return self.prob[item]