from collections import Counter, defaultdict
import csv

import socket
socket_name = socket.gethostname()
if any(name in socket_name for name in {"Gianni","vpn","Berenice"}):
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange
    
import regex as re
from os import makedirs, listdir
from os.path import isfile, isdir
from functools import reduce
import operator
from joblib import Parallel, delayed
import time

from tokenizers import normalizers
from datasets import Dataset,logging
logging.set_verbosity_error()
# from .syntagmatic import NormalizeSLG

from . import util
from .syntagmatic import Syntagmatic, NormalizeSLG, Normalize_w_spacesSLG # needed
from .paradigmatic import Paradigmatic

# TODO: Solve version as global variable
slg_version = "0.2.3"
    
class Vocabulary:
    
    def __init__(self,semiotic):
        
        #TODO: Is there another way than loading the corpus (or the semiotic) here?
        self.corpus = semiotic.corpus
        self.semiotic = semiotic
        
        self.name = semiotic.name
        self.path = semiotic.paths.vocabulary
        self.config = semiotic.config.vocabulary
        self.model = self.config.model
        self.cpu_count = semiotic.config.system.cpu_count
        
        self.merges = None
        self.encode = None
        self.freq = None
        self.alpha = None
        
        self.decode = None
        
        self.len = None
        self.freq_mass = None
        self.prob = None

        # Load HF normalizer
        
        # #TODO: The elif condition on the string SLG is sort of a hack (needed due to non standard declaration of custom normalizer). There should be a more elegant way

        config_normalizer = semiotic.config.syntagmatic.normalizer
        if config_normalizer != None:
            if isinstance(config_normalizer,list):
                normalizer = normalizers.Sequence(
                    [eval(f"normalizers.{norm}()") for norm in config_normalizer]
                    )
            elif "SLG" in config_normalizer:
                normalizer = eval(f"{config_normalizer}")
            else:
                normalizer = eval(f"normalizers.{config_normalizer}()")
        
            self.normalizer = normalizer.normalize_str
        
        else:
            # self.normalizer = lambda x: x
            self.normalizer = None


    def from_file(self,path = None):
        if path == None:
            path = self.path


        filenames = [(path / fn) for fn in ["merges.txt","vocab.json","freq.json","alpha.json"]]
        
        for filename in filenames:
            if not isfile(filename):
                return print(f"SLG [W]: {filename} does not exist.\nVocabulary will not be loaded from disk.\n")
        
        self.merges = [tuple(merge.split()) for merge in util.txt2list("merges",path)[1:]] # The first line needs to be stripped
        self.encode = util.json2dict("vocab",path)
        self.freq = util.json2dict("freq",path)
        self.alpha = util.json2dict("alpha",path)

        self.decode = {i:k for k,i in self.encode.items()}
        
        self.len = len(self.encode)
        self.freq_mass = sum(self.freq.values())
        self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}
        print(f"SLG [I]: Vocabulary loaded from disk")
    
        # TODO: implement better automatic loading of all files in ngram

        if isdir(self.path / "ngrams"):
            ngram_files = [f for f in listdir(self.path / "ngrams") if isfile(self.path / f"ngrams/{f}") and f[-4:]=="json"]

            self.ng1 = nGram(from_dict=self.alpha)
            if ngram_files!=[]:
                for f in ngram_files:
                    setattr(self, f"ng{f.split('.')[0]}", nGram(from_file = self.path / f"ngrams/{f}"))
                print(f"SLG [I]: nGrams loaded from disk ({ngram_files})")


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
        model = None,
        vocab_size = None,
        special_tokens = None,
        save = False,
        save_step = None,
        progress_bar = True,
        resume_merges = False,
        parallel = False,
        corpus_length = None,
        keep_in_memory = False, #HF argument in dataset shard and map
        ):

        if corpus == None:
            corpus = self.name
        
        if model == None:
            model = self.model
        
        if vocab_size == None:
            vocab_size = self.config.size

        if special_tokens == None:
            special_tokens = self.config.special_tokens

        if corpus_length == None:
            corpus_length = self.corpus.train_len

        if save == True and save_step != None:
            saveQ = True
            
            if not isdir(self.path):
                makedirs(self.path)
        else:
            saveQ = False

        if model == "SLG":
            def pre_process(corpus_chunk:Dataset, normalizer):

                # Normalize
                
                if normalizer == None:
                    chain_zip = corpus_chunk
                else:
                    # chain_zip = normalizer(corpus_chunk)
                    chain_zip = corpus_chunk.map(lambda sent: {"text": normalizer(sent["text"])}, keep_in_memory=keep_in_memory)

                chain_zip = "".join(chain_zip["text"])
                
                # Build list of pairs
                chain_zip = list(zip(chain_zip,chain_zip[1:]))
                # Create a lookup table of all the positions where a pair appears in a corpus
                pair_pos = defaultdict(set)
                for i,k in list(enumerate(chain_zip)):
                    pair_pos[k].add(i)
                # From the previous lookup table, create another lookup table of the frequency of each pair (given by the size of the set of its positions)
                pair_len = Counter()
                for k,pos in pair_pos.items():
                    pair_len[k] = len(pos)
                
                return (chain_zip, pair_pos, pair_len)

            def process_best_pair(chain_zip, pair_pos, best_pair):
                chain_zip_len = len(chain_zip)
                pair_len_delta = Counter()

                for i in pair_pos[best_pair]:
                    # Skip iteration if position corresponds to a modified set of positions during the iteration. This can happen if there is overlap of pairs, such as "000", where ("0","0") has itself as right pair. # Note that, due to unordered implementation of sets, this entails a lack of systematicity in overlapping cases: "000" can be counted randomly as ("00","0") or ("0","00").
                    if chain_zip[i]!=best_pair:
                        continue
                    ## merge best pair with left unit
                    left_pair_i = i-1
                    while left_pair_i>=0 and chain_zip[left_pair_i] == None: # if left pair is within chain limits but empty (= None) because already merged previously, shift to the left
                        left_pair_i -= 1
                    if left_pair_i>-1: # proceed only if a left pair was found on the left
                        # Remove from left pair positions, the current position (of the pair to be merged)
                        left_pair = chain_zip[left_pair_i]
                        # Skip update of left_pair position set if left_pair = best_pair, to avoid modification of iterating set. This can happen if there is overlap of pairs. No consequences on final result (right?) since right after the loop, the key corresponding to the best pair is deleted, and chain_zip is indeed updated so the problematic cases can be captured at the beginning of the loop.
                        if left_pair != best_pair:
                            pair_pos[left_pair].discard(left_pair_i)
                        new_pair = (left_pair[0],"".join(best_pair)) # construct new left pair
                        
                        # update the list of pairs
                        chain_zip[left_pair_i] = new_pair
                        # add new pair (if non existing) and its position to the pair_pos lookup table
                        pair_pos[new_pair].add(left_pair_i)
                        # update the counts in the pair_len lookuptable
                        pair_len_delta[left_pair] -= 1
                        pair_len_delta[new_pair] += 1


                    ## merge best pair with right unit.
                    # Code is symmetric to left_pair but on the right. Comments are omitted
                    right_pair_i = i+1
                    while right_pair_i<chain_zip_len and chain_zip[right_pair_i] == None:
                        right_pair_i += 1
                    if right_pair_i<chain_zip_len:
                        right_pair = chain_zip[right_pair_i]
                        if right_pair != best_pair:
                            pair_pos[right_pair].discard(right_pair_i)
                        new_pair = ("".join(best_pair), right_pair[1])

                        chain_zip[right_pair_i] = new_pair
                        pair_pos[new_pair].add(right_pair_i)
                        pair_len_delta[right_pair] -= 1
                        pair_len_delta[new_pair] += 1


                    # Empty best pair position in list of pairs
                    chain_zip[i] = None

                # Remove best pair from lookuptables
                del pair_pos[best_pair]

                return (chain_zip, pair_pos, pair_len_delta)

            def compute_freq(chain_zip):
                # Collect the first component of the pairs
                freq = [pair[0] for pair in chain_zip if pair != None]
                i = -1
                # Add the last component of the last pair
                while chain_zip[i]==None:
                    i -= 1
                freq.append(chain_zip[i][-1])
                # Count the units of the resulting (decoupled) chain list
                freq = Counter(freq)
                return freq
            

            if parallel:
                # TODO: The chunks limits could be improved (in particular, if corpus_length is very small compared to cpu_count, last chunks may be empty. It shouldn't be a problem for large corpus_length)
                
                print(f"SLG: Computing in parallel. CPU count: {self.cpu_count}")
                
                corpus_chunks = [self.corpus.train.shard(self.cpu_count,n, contiguous=True, keep_in_memory = keep_in_memory) for n in range(self.cpu_count)]

                with Parallel(n_jobs=self.cpu_count, require='sharedmem') as parallel_pool:
                    print("SLG [I]: Starting parallel jobs.")
                    print("SLG [I]: Normalize and jobs data...")
                    start = time.time()
                    jobs_data = parallel_pool(delayed(pre_process)(chunk,self.normalizer) for chunk in corpus_chunks)

                    pair_len_global = reduce(operator.add,[pair_len for chain_zip, pair_pos, pair_len in jobs_data])

                    # When pair_len_global has more than 1 max, the first encountered is chosen, introducing possible discrepancies between implementations (because each choice modifies global statistics). However, multiple max is less likely to appear in big corpora and relatively small vocabularies, and mostly at the tail of vocabularies (ie. low frequencies), so the impact of this divergence is expected to be marginal.
                    best_pair, best_pair_len = max(pair_len_global.items(), key=operator.itemgetter(1))
                    
                    merges = [" ".join(best_pair)]
                    print(f"... computed in {time.time()-start} secs.\n")

                    print("SLG [I]: Build alphabet...")
                    start = time.time()
                    alphabet = Counter()
                    for (l,r),v in pair_len_global.items():
                        alphabet[l] += v
                    # In extreme cases, right characters of pairs might not be left characters. If there are such chars, they're added with freq 1
                    left_out_chars = {r for l,r in pair_len_global.keys()}-alphabet.keys()
                    if len(left_out_chars)>0:
                        print(f"SLG: Adding characters: {left_out_chars}")
                        for char in left_out_chars:
                            alphabet[char] += 1
                    print(f"... computed in {time.time()-start} secs.\n")

                    alpha_len = len(alphabet)
                    special_tokens_len = 0 if special_tokens == None else len(special_tokens)
                    
                    print(f"SLG: Alphabet Size: {alpha_len}")
                    print(f"SLG: Special Tokens Size: {special_tokens_len}")
                    
                    if vocab_size<0:
                        voc_final_length = alpha_len + special_tokens_len + abs(vocab_size)
                    else:
                        voc_final_length = vocab_size

                    delta_voc = voc_final_length - alpha_len - special_tokens_len

                    print(f"SLG: Terms to compute: {delta_voc}\n")

                    print("SLG: Enter loop")

                    t = trange(delta_voc, disable = not progress_bar)
                    for _ in t:
                        t.set_description(f"Pair: {best_pair}, {best_pair_len}")
                        t.refresh()

                        jobs_data = parallel_pool(delayed(process_best_pair)(chain_zip, pair_pos, best_pair) for chain_zip, pair_pos, pair_len_delta in jobs_data)

                        for chain_zip, pair_pos, pair_len_delta in jobs_data:
                            pair_len_global.update(pair_len_delta)

                        # Remove best_pair from pair_len
                        del pair_len_global[best_pair]

                        # When pair_len_global has more than 1 max, the first encountered is chosen, introducing possible discrepancies between implementations (because each choice modifies global statistics). However, multiple max is less likely to appear in big corpora and relatively small vocabularies, and mostly at the tail of vocabularies (ie. low frequencies), so the impact of this divergence is expected to be marginal.
                        best_pair, best_pair_len = max(pair_len_global.items(), key=operator.itemgetter(1))

                        merges.append(" ".join(best_pair))

                        if saveQ == True:
                            voc_partial_len = alpha_len + special_tokens_len + _ + 1
                            if voc_partial_len % save_step == 0 and voc_partial_len != voc_final_length:

                                print("SLG [I]: Saving intermediate results...")
                                start = time.time()
                                freqs = parallel_pool(delayed(compute_freq)(chain_zip) for chain_zip, pair_pos, pair_len_delta in jobs_data)
                                freq = reduce(operator.add, freqs)

                                vocabulary = freq.most_common()
                                
                                if special_tokens != None:
                                    vocabulary = vocabulary + [(token,0) for token in special_tokens]
                                
                                self.merges = merges
                                self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
                                self.freq = dict(vocabulary)
                                self.alpha = dict(alphabet.most_common())
                                step_path = self.path /  "checkpoints/" + str(voc_partial_len)
                                self.save(step_path)
                                print(f"... computed in {time.time()-start} secs.")
                                print(f"SLG [I]: Intermediate vocabulary saved to {step_path}\n")

                    print("SLG [I]: Compute freq...")
                    start = time.time()
                    freqs = parallel_pool(delayed(compute_freq)(chain_zip) for chain_zip, pair_pos, pair_len_delta in jobs_data)
                    freq = reduce(operator.add, freqs)
                    print(f"... computed in {time.time()-start} secs.\n")
            
            else:
                #TODO: Sequential computing not completely tested
                print("SLG [I]: Computing sequentially")
                print("SLG [I]: Normalize and jobs data...")
                start = time.time()
                corpus_chain = self.corpus.train[:corpus_length]
                chain_zip, pair_pos, pair_len_global = pre_process(corpus_chain,self.normalizer)

                # When pair_len_global has more than 1 max, the first encountered is chosen, introducing possible discrepancies between implementations (because each choice modifies global statistics). However, multiple max is less likely to appear in big corpora and relatively small vocabularies, and mostly at the tail of vocabularies (ie. low frequencies), so the impact of this divergence is expected to be marginal.
                best_pair, best_pair_len = max(pair_len_global.items(), key=operator.itemgetter(1))
                
                merges = [" ".join(best_pair)]
                print(f"... computed in {time.time()-start} secs.\n")

                print("SLG [I]: Build alphabet...")
                start = time.time()
                alphabet = Counter()
                for (l,r),v in pair_len_global.items():
                    alphabet[l] =+ v
                # In extreme cases, right characters of pairs might not be left characters. If there are such chars, they're added with freq 1
                left_out_chars = {r for l,r in pair_len_global.keys()}-alphabet.keys()
                if len(left_out_chars)>0:
                    print(f"SLG: Adding characters: {left_out_chars}")
                    for char in left_out_chars:
                        alphabet[char] =+ 1
                print(f"... computed in {time.time()-start} secs.\n")

                alpha_len = len(alphabet)
                special_tokens_len = 0 if special_tokens == None else len(special_tokens)
                
                print(f"SLG: Alphabet Size: {alpha_len}")
                print(f"SLG: Special Tokens Size: {special_tokens_len}")
                
                if vocab_size<0:
                    voc_final_length = alpha_len + special_tokens_len + abs(vocab_size)
                else:
                    voc_final_length = vocab_size

                delta_voc = voc_final_length - alpha_len - special_tokens_len
                
                print(f"SLG: Terms to compute: {delta_voc}\n")

                print("SLG: Enter loop")

                t = trange(delta_voc, disable = not progress_bar)
                for _ in t:
                    t.set_description(f"Pair: {best_pair}, {best_pair_len}")
                    t.refresh()

                    chain_zip, pair_pos, pair_len_delta = process_best_pair(chain_zip, pair_pos, best_pair)

                    # Remove best_pair from pair_len
                    pair_len_global.update(pair_len_delta)

                    del pair_len_global[best_pair]

                    # When pair_len_global has more than 1 max, the first encountered is chosen, introducing possible discrepancies between implementations (because each choice modifies global statistics). However, multiple max is less likely to appear in big corpora and relatively small vocabularies, and mostly at the tail of vocabularies (ie. low frequencies), so the impact of this divergence is expected to be marginal.
                    best_pair, best_pair_len = max(pair_len_global.items(), key=operator.itemgetter(1))

                    merges.append(" ".join(best_pair))
                    # print(f"... computed in {time.time()-start} secs.\n")

                    if saveQ == True:
                        voc_partial_len = alpha_len + special_tokens_len + _ + 1
                        if voc_partial_len % save_step == 0 and voc_partial_len != voc_final_length:

                            print("SLG: Saving intermediate results...")
                            start = time.time()
                            freq = compute_freq(chain_zip)

                            vocabulary = freq.most_common()
                            
                            if special_tokens != None:
                                vocabulary = vocabulary + [(token,0) for token in special_tokens]
                            
                            self.merges = merges
                            self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
                            self.freq = dict(vocabulary)
                            self.alpha = dict(alphabet.most_common())
                            step_path = self.path / str(voc_partial_len)
                            self.save(step_path)
                            print(f"... computed in {time.time()-start} secs.")
                            print(f"SLG: Intermediate vocabulary saved to {step_path}\n")
                
                print("SLG [I]: Compute freq...")
                start = time.time()
                freq = compute_freq(chain_zip)
                print(f"... computed in {time.time()-start} secs.\n")

        elif model == "SLG_WL":

            freq = Counter(" ".join(self.corpus.train["text"]).split())
            alphabet = Counter(" ".join(self.corpus.train["text"]))
            merges = []
        
        else:
            return f"Vocabulary building model ({model}) not recognized"


        vocabulary = freq.most_common()
        
        if special_tokens != None:
            vocabulary = [(token,0) for token in special_tokens] + vocabulary
        
        self.merges = merges
        self.encode = {k:i for i,(k,v) in enumerate(vocabulary)}
        self.freq = dict(vocabulary)
        self.alpha = dict(alphabet.most_common())

        self.decode = {i:k for k,i in self.encode.items()}
        
        self.len = len(vocabulary)     
        self.freq_mass = sum(self.freq.values())
        self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

        print("SLG [I]: Vocabulary built\n")
        
        if save == True:
            self.save()
            print(f"SLG [I]: Vocabulary saved to {self.path}\n")
        
        self.semiotic.syntagmatic = Syntagmatic(self.semiotic)
        self.semiotic.paradigmatic = Paradigmatic(self.semiotic)
        print("SLG [I]: Syntagmatic and Paradigmatic updated with the new vocabulary\n")

    def save(self, path = None):
        
        if path == None:
            path = self.path
            
        version_stamp = f"#version: {slg_version} - Built by `semiolog`"
        
        util.list2txt([version_stamp]+self.merges,"merges", path)
        util.dict2json(self.encode,"vocab", path)
        util.dict2json(self.freq,"freq", path)
        util.dict2json(self.alpha,"alpha", path)
        
class nGram():
    def __init__(self, from_file = None, from_dict = None):

        if isinstance(from_dict,dict):
            self.freq = from_dict
        
        elif isfile(from_file):
            self.freq = util.load_file(from_file)
        else:
            raise Exception(f"SLG [E]: No valid dictionnary or filename provided.")

        self.keys = list(self.freq.keys())
        self.encode = {t:i for i,t in enumerate(self.keys)}
        self.decode = {i:t for i,t in enumerate(self.keys)}

        # self.prob = util.normalize_dict(self.freq)

    def extract_ngrams(text, n, count = False):

        if count:
            n_grams = Counter(zip(*[text[i:] for i in range(n)]))
        else:
            n_grams = list(zip(*[text[i:] for i in range(n)]))

        return n_grams

    def build(
        corpus=None,
        n=2,
        thres = 1,
        str_normalizer = None,
        parallel = False,
        keep_in_memory = False,
        cpu_count = 8,
        ):
        
        print("SLG [W]: This feature is not fully implemented/tested yet")
        #TODO: implement automatic loading of corpus if corpus==None
        #TODO: Dynamic cpu_count
        #TODO: implement save option

        cpu_count = 8

        def count_ngrams(corpus,n,str_normalizer):
            """
            Corpus needs to be a list of strings
            """
            corpus = " ".join(corpus)

            print(f"SLG: Counting {n}-Grams...")
            
            ngrams = Counter()
            for ng in tqdm(zip(*[corpus[i:] for i in range(n)]),total=(len(corpus)-n)):
                ngrams[ng]+=1

            return ngrams

        if str_normalizer != None:
            print(f"SLG: Normalizing Corpus...")
            corpus_norm = []
            for sent in tqdm(corpus["text"]):
                corpus_norm.append(str_normalizer(sent))

            corpus = Dataset.from_dict({"text":corpus_norm})
            del(corpus_norm)

        if parallel:
            corpus_chunks = [corpus.shard(cpu_count, n, contiguous=True, keep_in_memory = keep_in_memory) for n in range(cpu_count)]

            with Parallel(n_jobs=cpu_count, require='sharedmem') as parallel_pool:
                jobs_data = parallel_pool(delayed(count_ngrams)(chunk["text"],n,str_normalizer) for chunk in corpus_chunks)

                ngrams = reduce(operator.add,jobs_data)

        else:
        
            ngrams = count_ngrams(corpus=corpus["text"],n=n,str_normalizer=str_normalizer)

        ngrams = {"".join(tup) : freq for tup, freq in ngrams.most_common() if " " not in tup and freq>=thres}

        # if save:
        #     slg.util.save_file(ngrams,semiotic.paths.vocabulary / f"ngrams/{n}.json")

        return ngrams
    

    def __repr__(self) -> str:
        return f"nGram({self.freq})"

    # def __getitem__(self, item):
    #      return self.freq.items()[item]


# class nGram(Vocabulary):
#     def __init__(self, filename = None, special_tokens = None):

#         if filename != None:
                
#             with open(filename, "r") as f:
#                 csv_reader = csv.reader(f)
#                 voc = Counter()
#                 for line in csv_reader:
#                     voc[tuple(line[:2])] = int(line[-1])
#                 voc = dict(voc.most_common())

#             self.filename = filename
#             self.len = len(voc)
#             self.freq = voc
#             self.freq_mass = sum(voc.values())
#             self.prob = {k:v/self.freq_mass for k,v in self.freq.items()}

#             self.encode = {k:i for i,(k,v) in enumerate(voc.items())}
#             self.decode = {i:k for k,i in self.encode.items()}
#         else:
#             pass

#     def __repr__(self) -> str:
#         return f"nGram({self.freq})"

#     def __str__(self) -> str:
#         return str(self.freq)

#     def __getitem__(self, item):
#          return self.prob[item]