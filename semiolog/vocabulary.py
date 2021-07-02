from collections import Counter
import csv
from typing import Union, Iterable, Dict, Any
from pathlib import Path

class Vocabulary:
    def __init__(self,filename = None,special_tokens = None): # TODO: Handle special tokens
        if filename != None:
                
            with open(filename, "r") as f:
                csv_reader = csv.reader(f)
                voc = Counter()
                for line in csv_reader:
                    voc[line[0]] = int(line[1])
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
        return f"Voc({self.freq})"

    def __str__(self) -> str:
        return str(self.freq)

    def __getitem__(self, item):
         return self.prob[item]

    def head(self,size=10):
        pass

    def alphabetic(self):
        pass

    def keys(self):
        pass

    def values(self):
        pass


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
    

from collections import Counter
from semiolog import util_g
import functools
import time
import operator
import regex as re
from tqdm.notebook import tqdm, trange

def find_best_pair(chain_spaced):
    pairs = Counter()
    pre_units = chain_spaced.split()
    for i in range(len(pre_units) - 1):
        pairs[pre_units[i], pre_units[i + 1]] += 1
    return pairs.most_common()[0][0]

def agglutinate_chain(pair, chain_spaced):
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_chain = p.sub("".join(pair), chain_spaced)
    return new_chain

def count_units_in_chain(chain_spaced):
    vocab = Counter()
    chain_units = chain_spaced.split()
    for unit in chain_units:
        vocab[unit] += 1
    return dict(vocab.most_common())

def build_vocab(
    chain:str,
    voc_length: int,
    inter_save = [],
    save_finalQ = True,
    filename = "voc_default",
    resumeQ = False,
    corpus_name = "corp_deefault",
    courpus_length = None,
    printQ = True,
):
    if resumeQ:
        initial_chain = chain.replace(" ", "")
    else:
        initial_chain = chain
    spaced_chain = " ".join(chain)
    alpha = count_units_in_chain(spaced_chain)
    alpha_len = len(alpha)
    print(f"Length of initial alphabet: {alpha_len}")
    if resumeQ:
        print("Resuming chain...")
        spaced_chains = chain
        print("Chain resumed")
    if voc_length > 0:
        print("Enter loop")
        for i in trange(voc_length):
            start = time.perf_counter()
            best_pair = find_best_pair(spaced_chain)
            spaced_chain = agglutinate_chain(best_pair, spaced_chain)
            finish = time.perf_counter()
            if printQ:
                print(f"{i}: {''.join(best_pair)} - Computed in {round(finish - start,2)} secs")
            # if i in inter_save:
            #     vocab = count_units_in_chain(spaced_chain)
            #     print(
            #         f"Saving intermediate result. Vocabulary length: {len(vocab)-alpha_len}"
            #     )
            #     output_file = (
            #         f"voc_A_{corpus_name}_{len(vocab)-alpha_len}_inter"
            #     )
            #     util_g.dict2csv(vocab, output_file, paths.vocabularies)

            #     output_resume = f"{corpus_name}_resume"
            #     util_g.str2txt(spaced_chain, output_resume, paths.scratch)
            #     util_g.str2txt(
            #         f"Input corpus: {corpus_name}\nCorpus Lenght: {courpus_length}\nParallel: {parallelQ}\nNumbers of Cores: {n_cores}\nVocabulary length (so far): {len(vocab)-alpha_len}",
            #         f"{corpus_name}_resume_info",
            #         paths.scratch,
            #     )
    print("Collecting frequencies of terms")
    vocab = count_units_in_chain(spaced_chain)
    # if save_finalQ:
    #     util_g.dict2csv(vocab, filename, paths.vocabularies)
    return vocab

def find_best_pair_par(chain_spaced):
    pairs = Counter()
    pre_units = chain_spaced.split()
    for i in range(len(pre_units) - 1):
        pairs[pre_units[i], pre_units[i + 1]] += 1
    return dict(
        pairs.most_common()[:100]
    )  # Looking only on top 100 pre_units of each par list, for efficiency

def build_vocab_par(
    chain: str,
    voc_length: int,
    n_cores=4,
    inter_save=[],
    resumeQ = False,
    save_finalQ=False,
    filename = "voc_p_default",
    corpus_name = "corp_deefault",
    courpus_length = None,
    ):
    if resumeQ:
        initial_chain = chain.replace(' ','').replace('\n','')
    else:
        initial_chain = chain
    print("Spacing chain...")
    spaced_chains = [" ".join(chain_part) for chain_part in util_g.partition(initial_chain, n_cores)]
    print(f"Chain spaced")
    alphas = util_g.multiprocessing(count_units_in_chain, spaced_chains)
    alpha = dict(functools.reduce(operator.add, [Counter(voc) for voc in alphas]).most_common())
    apha_len = len(alpha)
    print(f'Length of initial alphabet: {apha_len}')
    if resumeQ:
        print("Resuming chain...")
        spaced_chains = chain.split('\n')
        print('Chain resumed')
    if voc_length > 0:
        print("Enter loop")
        for i in range(voc_length):
            start = time.perf_counter()
            top_pairs_par = util_g.multiprocessing(find_best_pair_par, spaced_chains)
            best_pair = functools.reduce(
                operator.add, [Counter(top_pairs) for top_pairs in top_pairs_par]
            ).most_common()[0][0]
            spaced_chains = util_g.multiprocessing(
                functools.partial(agglutinate_chain, best_pair), spaced_chains
            )
            finish = time.perf_counter()
            print(
                f"{i}: {''.join(best_pair)} - Computed in {round(finish - start,2)} secs"
            )
            # if i in inter_save:
            #     vocabs = util_g.multiprocessing(count_units_in_chain, spaced_chains)
            #     vocab = dict(
            #         functools.reduce(
            #             operator.add, [Counter(voc) for voc in vocabs]
            #         ).most_common()
            #     )
            #     print(f"Saving intermediate result. Vocabulary length: {len(vocab)-apha_len}")
            #     output_file = f"voc_A_{corpus_name}_p_{len(vocab)-apha_len}_inter"
            #     util_g.dict2csv(vocab, output_file, paths.vocabularies)

            #     output_resume = f"{corpus_name}_p_resume"
            #     util_g.str2txt('\n'.join(spaced_chains), output_resume, paths.scratch)
            #     util_g.str2txt(f'Input corpus: {corpus_name}\nCorpus Lenght: {courpus_length}\nParallel: {parallelQ}\nNumbers of Cores: {n_cores}\nVocabulary length (so far): {len(vocab)-apha_len}',f"{corpus_name}_p_resume_info",paths.scratch)
    print('Collecting frequencies of terms')
    vocabs = util_g.multiprocessing(count_units_in_chain, spaced_chains)
    vocab = dict(
        functools.reduce(operator.add, [Counter(voc) for voc in vocabs]).most_common()
    )
    # if save_finalQ:
    #     util_g.dict2csv(vocab, filename, paths.vocabularies)
    return vocab

def build_vocabulary(
    chain,
    printQ = True,
    parallelQ = True,
    n_cores = 4,
    voc_len = 10,
    inter_results = False,
    resumeQ = False,
    save_finalQ = False,
    filename = "voc_default",
    corpus_name = "corp_deefault",
    courpus_length = None,
):
    start = time.perf_counter()
    print(f"Building Vocabulary")
    if printQ:
        print("Print is on")
    else:
        print("Print is off")
    if parallelQ:
        print(f"Method: Parallel - N° of Cores: {n_cores}")
        voc = build_vocab_par(chain, voc_len, n_cores, inter_save=inter_results, save_finalQ=save_finalQ, resumeQ = resumeQ, filename = filename, corpus_name = corpus_name, courpus_length = courpus_length)
    else:
        print(f"Method: Sequential")
        voc = build_vocab(chain, voc_len, inter_save=inter_results, save_finalQ=save_finalQ,resumeQ = resumeQ, filename = filename, corpus_name = corpus_name, courpus_length = courpus_length, printQ=printQ)
    finish = time.perf_counter()
    print(f"Vocabulary built in {round(finish - start,2)} secs")
    return voc