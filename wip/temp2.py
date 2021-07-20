####### TEST BUILD VOC, EFFACER #######
import os
os.chdir("../")

import semiolog as slg
from semiolog.util import multiprocessing, multiprocessing_tqdm


import regex as re
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from tqdm.notebook import tqdm, trange
from functools import partial
from functools import reduce
import operator

def findall_contexts(chain,best_pair_string,re_voc_l,re_voc_r):
    contexts = re.findall(re_voc_l+best_pair_string+re_voc_r, chain, overlapped=True)
    return contexts


def find_best_pair(chain_list):
    pair_count = Counter()
    for pair in list(zip(chain_list, chain_list[1:])):
        pair_count[pair] += 1
    return pair_count

def chain_list_alpha(normalizer, corpus_sents, progress_bar = False):

    chain_list = []
    alphabet = Counter()
    
    if progress_bar:
        for sent in tqdm(corpus_sents, desc="Normalize & Alphabet"):
            sent = normalizer.normalize(sent)
            sent = list(sent)
            if sent !=[]:
                chain_list += sent
                alphabet.update(Counter(sent))
    else:
        for sent in corpus_sents:
            sent = normalizer.normalize(sent)
            sent = list(sent)
            if sent !=[]:
                chain_list += sent
                alphabet.update(Counter(sent))
                
    return chain_list, alphabet

def build_nb(
    corpus = None,
    voc_final_length = -30,
    # save = False,
    # save_step = None,
    # progress_bar = True,
    # resume_merges = False,
    parallel = True,
    sparse = True,
    sparse_mode = "csr",
    cpu_count = 4,
    corpus_length = None,
    normalizer = None,
):
    def agglutinate_chain(pair, cl_chain):
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        cl_chain = p.sub("".join(pair), cl_chain)
        return cl_chain

    def extract_drc(pairs, encoder: dict):
        data = []
        rows = []
        columns = []
        for (r,c),d in pairs:
            data.append(d)
            rows.append(encoder[r])
            columns.append(encoder[c])
        return data, rows, columns

    def parallel_chain(chain, n_of_parts, overlap = 0):
        """
        Breaks the chain in n chunks to compute best pair of terms. Chunks are overlapping by one term, so as no pair of terms is lost due to the break.
        """
        if not isinstance(chain,list):
            chain = list(chain)
        chunk_size = int(len(chain) / n_of_parts)+1
        for i in range(0, len(chain), chunk_size):
            yield chain[i : i + chunk_size + overlap]

    def separate_chain(chain, n_of_parts, best_pair: list):
        """
        Separate a chain (in list form) for parallel processing of regex findall of pair, taking care that the cuts of the chunks don't fall in the neiborhood of the pair, affecting the final counts
        """
        chunk_size = int(len(chain) / n_of_parts)+1
        b = 0
        n = chunk_size
        chain_len = len(chain)
        for i in range(n_of_parts):
            n = (i+1)*chunk_size
            if chain_len > n:
                while chain[n-2:n] == best_pair or chain[n-1:n+1] == best_pair:
                    n = n+1
            yield ("[SEP_i] " if i!=0 else "") + " ".join(chain[b:n]) + (" [SEP_i]" if i!=n_of_parts-1 else "")
            b = n-1
        
        

    
    if parallel:
        
        par_corpus = parallel_chain(corpus[:corpus_length], cpu_count)

        result = multiprocessing_tqdm(partial(chain_list_alpha, normalizer), par_corpus, cores=cpu_count, desc="Normalize & Alphabet")
        
        chain_list = []
        alphabet = Counter()
        for chain_l, alpha in result:
            chain_list += chain_l
            alphabet += alpha
            
    else:
        chain_list, alphabet = chain_list_alpha(normalizer, corpus[:corpus_length], progress_bar=True)

    cl_chain = "[SEP] "+" ".join(chain_list)+" [SEP]"
    encode = {k:i for i,(k,v) in enumerate(alphabet.most_common())}
    decode = {i:k for k,i in encode.items()}
    new_i = len(encode)
    if parallel:
        
        par_chain = parallel_chain(chain_list, cpu_count, overlap=1)
        
        result = multiprocessing(find_best_pair, par_chain, cores=cpu_count) 
                            
        pairs = reduce(operator.add, result)
        pairs = pairs.most_common()
        
    else:
        pairs = find_best_pair(chain_list).most_common()
    if voc_final_length<0:
        voc_final_length = new_i + abs(voc_final_length)
        
    if sparse:
        data, rows, columns = extract_drc(pairs,encode)
        voc_matrix = coo_matrix((np.array(data), (np.array(rows),np.array(columns))), shape=(voc_final_length, voc_final_length), dtype=int)

    else:
        voc_matrix = np.zeros((voc_final_length, voc_final_length), dtype=int)
        for (row,column),value in pairs:
            voc_matrix[encode[row], encode[column]] = value
    merges = []
    delta_voc = voc_final_length - new_i
    best_pair = "init"
    pair_count = "---"

    t = trange(delta_voc) #, disable = not progress_bar)

    for _ in t:
        t.set_description(f"Pair: {best_pair}, {pair_count}")
        t.refresh()

        if sparse:
            max_i = voc_matrix.data.argmax()
            pair_row = voc_matrix.row[max_i]
            pair_col = voc_matrix.col[max_i]
            pair_count = voc_matrix.data[max_i]
        else:
            pair_row,pair_col = np.unravel_index(np.argmax(voc_matrix, axis=None), voc_matrix.shape)
            pair_count = voc_matrix[pair_row,pair_col]
        
        if pair_count == 0:
            break

        best_pair = (decode[pair_row], decode[pair_col])
        best_pair_string = " ".join(best_pair)
        merges.append(best_pair_string)
        best_pair_string_voc = "".join(best_pair)
        re_voc_l = "("+"|".join([" "+k+" " for k in encode.keys()]+["\[SEP\] ","\[SEP_i\] "])+")"
        re_voc_r = "("+"|".join([" "+k+" " for k in encode.keys()]+[" \[SEP\]"," \[SEP_i\]"])+")"
        if parallel:
            result = multiprocessing(
                partial(findall_contexts,best_pair_string=best_pair_string,re_voc_l=re_voc_l,re_voc_r=re_voc_r),
                separate_chain(cl_chain.split(), cpu_count, list(best_pair)),
                cores = cpu_count
                )
            merge_context = reduce(operator.add, result)
        else:
            merge_context = re.findall(re_voc_l+best_pair_string+re_voc_r, cl_chain, overlapped=True)
        merge_context_count_l = Counter()
        merge_context_count_r = Counter()
        for l,r in merge_context:
            if "[SEP]" not in l:
                merge_context_count_l[encode[l.strip()]] += 1
            if "[SEP]" not in r:
                merge_context_count_r[encode[r.strip()]] += 1
        
        if sparse:
            # Convert matrix to CSR or LIL, for item attribution and arithmetic 
            if sparse_mode == "csr":
                voc_matrix = voc_matrix.tocsr()
            else:
                voc_matrix = voc_matrix.tolil()
        
        for row,key in merge_context_count_l.items():
            voc_matrix[row,new_i] = key
            
        for column,key in merge_context_count_r.items():
            voc_matrix[new_i,column] = key

        # Correct previous counts
        
        # compute #(l,r)-(l,r)
        pair_pair_count = len(re.findall(" "+best_pair_string+" "+best_pair_string+" ", cl_chain, overlapped=False))
        # remove #(l,r)-(l,r) from (l,r)-l
        voc_matrix[new_i,pair_row] -= pair_pair_count
        # remove #(l,r)-(l,r) from r-(l,r)
        voc_matrix[pair_col,new_i] -= pair_pair_count
        # remove #(l,r)-(l,r) from r-l
        voc_matrix[pair_col,pair_row] -= pair_pair_count
        # substract (l,r)- from r-
        voc_matrix[pair_col,:new_i] -= voc_matrix[new_i,:new_i]
        # substract -(l,r)- from -l
        voc_matrix[:new_i,pair_row] -= voc_matrix[:new_i,new_i]
        
        # set l-r to 0
        voc_matrix[pair_row,pair_col] = 0
        # register #(l,r)-(l,r)
        voc_matrix[new_i,new_i] = pair_pair_count
        
        if sparse:
            # Convert matrix back to COO, to restart the loop
            voc_matrix = voc_matrix.tocoo()
        
        best_pair_string_voc = "".join(best_pair)
        encode[best_pair_string_voc] = new_i
        decode[new_i] = best_pair_string_voc
        new_i += 1
        cl_chain = agglutinate_chain(best_pair_string.split(),cl_chain)


    if sparse:
        freq_values = voc_matrix.sum(axis=1).T.tolist()[0]
    else:
        freq_values = voc_matrix.sum(axis=1).T.tolist()
    vocabulary = {decode[i]:v for i,v in enumerate(freq_values) if v>0} # Make sure dimension of matrix and size of voc coincide
    vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
    
    return merges, vocabulary


from pyinstrument import Profiler
import sys

semiotic = slg.Cenematic("fr_wiki")

norm_seq = semiotic.config.vocabulary.normalizer
normalizer = slg.syntagmatic.tokenizer.normalizers.Sequence(norm_seq)

profile = Profiler()
profile.start()

merges, vocabulary = build_nb(
    semiotic.corpus.train,
    normalizer = normalizer,
    voc_final_length = -10)

profile.stop()
print(profile.output_text(unicode=True, color=True))