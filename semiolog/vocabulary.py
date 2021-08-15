from collections import Counter, defaultdict
import csv

import socket
socket_name = socket.gethostname()
if any(name in socket_name for name in {"Gianni","vpn"}):
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange
    
import regex as re
from os import makedirs
from os.path import isfile, isdir
from functools import reduce
import operator
from joblib import Parallel, delayed
import time

from . import util
from .syntagmatic import tokenizer # needed

# TODO: Solve version as global variable
slg_version = "0.1.1"
    
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

        if isinstance(self.config.normalizer,list):
            self.normalizer = eval(
                f"tokenizer.normalizers.Sequence({self.config.normalizer})"
                )
        else:
            self.normalizer = eval(
                f"tokenizer.normalizers.{util.if_none_disable(self.config.normalizer)}"
            )


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
        corpus_length = None
        ):

        if corpus == None:
            corpus = self.name
        
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

        def pre_process(corpus_chunk, normalizer):
            # Normalize
            chain_zip = normalizer(corpus_chunk)
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
            chunksize = int(corpus_length/self.cpu_count)+1

            corpus_chunks = ["".join(self.corpus.train[i*chunksize:i*chunksize+chunksize]) for i in range(0,self.cpu_count)]

            with Parallel(n_jobs=self.cpu_count, require='sharedmem') as parallel_pool:
                print("Computing in parallel")
                print("Normalize and jobs data...")
                start = time.time()
                jobs_data = parallel_pool(delayed(pre_process)(chunk,self.normalizer.normalize) for chunk in corpus_chunks)

                pair_len_global = reduce(operator.add,[pair_len for chain_zip, pair_pos, pair_len in jobs_data])

                # When pair_len_global has more than 1 max, the first encountered is chosen, introducing possible discrepancies between implementations (because each choice modifies global statistics). However, multiple max is less likely to appear in big corpora and relatively small vocabularies, and mostly at the tail of vocabularies (ie. low frequencies), so the impact of this divergence is expected to be marginal.
                best_pair, best_pair_len = max(pair_len_global.items(), key=operator.itemgetter(1))
                
                merges = [" ".join(best_pair)]
                print(f"... computed in {time.time()-start} secs.\n")

                print("Build alphabet...")
                start = time.time()
                alphabet = Counter()
                for (l,r),v in pair_len_global.items():
                    alphabet[l] += v
                # In extreme cases, right characters of pairs might not be left characters. If there are such chars, they're added with freq 1
                left_out_chars = {r for l,r in pair_len_global.keys()}-alphabet.keys()
                if len(left_out_chars)>0:
                    print(f"Adding characters: {left_out_chars}")
                    for char in left_out_chars:
                        alphabet[char] += 1
                print(f"... computed in {time.time()-start} secs.\n")

                alpha_len = len(alphabet)
                special_tokens_len = 0 if special_tokens == None else len(special_tokens)
                
                print(f"Alphabet Size: {alpha_len}")
                print(f"Special Tokens Size: {special_tokens_len}")
                
                if vocab_size<0:
                    voc_final_length = alpha_len + special_tokens_len + abs(vocab_size)
                else:
                    voc_final_length = vocab_size

                delta_voc = voc_final_length - alpha_len - special_tokens_len

                print(f"Terms to compute: {delta_voc}\n")

                print("Enter loop")

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

                            print("Saving intermediate results...")
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
                            step_path = self.path / str(voc_partial_len)
                            self.save(step_path)
                            print(f"... computed in {time.time()-start} secs.")
                            print(f"Intermediate vocabulary saved to {step_path}\n")

                print("Compute freq...")
                start = time.time()
                freqs = parallel_pool(delayed(compute_freq)(chain_zip) for chain_zip, pair_pos, pair_len_delta in jobs_data)
                freq = reduce(operator.add, freqs)
                print(f"... computed in {time.time()-start} secs.\n")
        
        else:
            #TODO: Sequential computing not completely tested
            print("Computing sequentially")
            print("Normalize and jobs data...")
            start = time.time()
            corpus_chain = "".join(self.corpus.train[:corpus_length])
            chain_zip, pair_pos, pair_len_global = pre_process(corpus_chain,self.normalizer.normalize)

            # When pair_len_global has more than 1 max, the first encountered is chosen, introducing possible discrepancies between implementations (because each choice modifies global statistics). However, multiple max is less likely to appear in big corpora and relatively small vocabularies, and mostly at the tail of vocabularies (ie. low frequencies), so the impact of this divergence is expected to be marginal.
            best_pair, best_pair_len = max(pair_len_global.items(), key=operator.itemgetter(1))
            
            merges = [" ".join(best_pair)]
            print(f"... computed in {time.time()-start} secs.\n")

            print("Build alphabet...")
            start = time.time()
            alphabet = Counter()
            for (l,r),v in pair_len_global.items():
                alphabet[l] =+ v
            # In extreme cases, right characters of pairs might not be left characters. If there are such chars, they're added with freq 1
            left_out_chars = {r for l,r in pair_len_global.keys()}-alphabet.keys()
            if len(left_out_chars)>0:
                print(f"Adding characters: {left_out_chars}")
                for char in left_out_chars:
                    alphabet[char] =+ 1
            print(f"... computed in {time.time()-start} secs.\n")

            alpha_len = len(alphabet)
            special_tokens_len = 0 if special_tokens == None else len(special_tokens)
            
            print(f"Alphabet Size: {alpha_len}")
            print(f"Special Tokens Size: {special_tokens_len}")
            
            if vocab_size<0:
                voc_final_length = alpha_len + special_tokens_len + abs(vocab_size)
            else:
                voc_final_length = vocab_size

            delta_voc = voc_final_length - alpha_len - special_tokens_len
            
            print(f"Terms to compute: {delta_voc}\n")

            print("Enter loop")

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

                        print("Saving intermediate results...")
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
                        print(f"Intermediate vocabulary saved to {step_path}\n")
            
            print("Compute freq...")
            start = time.time()
            freq = compute_freq(chain_zip)
            print(f"... computed in {time.time()-start} secs.\n")

        vocabulary = freq.most_common()
        
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