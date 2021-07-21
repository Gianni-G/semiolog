# GENERAL FUNCTIONS #
#####################

# Load Modules

import os,sys
import math
import concurrent.futures
import csv
import json
import itertools
import ast
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import socket
socket_name = socket.gethostname()
if any(name in socket_name for name in {"Gianni","vpn"}):
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange

# Definitions of Functions

def chunk_size(total_n, chunks_n):
    return math.ceil(total_n / chunks_n)

def partition(lst, n_of_parts):
    """Partitions list in n parts."""
    if not isinstance(lst,list):
        lst = list(lst)
    chunk_size = math.ceil(len(lst) / n_of_parts)
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

# head and chunks taken from https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it

def head(iterable, max=10):
    first = next(iterable)      # raise exception when depleted
    def head_inner():
        yield first             # yield the extracted first element
        for cnt, el in enumerate(iterable):
            yield el
            if cnt + 1 >= max:  # cnt + 1 to include first
                break
    return head_inner()

def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))
        
def multiprocessing(func, args, chunksize=1, cores=None):
    with concurrent.futures.ProcessPoolExecutor(cores) as executor:
        result = executor.map(func, args, chunksize=chunksize)
    return list(result)

def multithreading(func, args, chunksize=1,cores=None):
    with concurrent.futures.ThreadPoolExecutor(cores) as executor:
        result = executor.map(func, args, chunksize=chunksize)
    return list(result)


# Adapted Dans Shiebler: from http://danshiebler.com/2016-09-14-parallel-progress-bar/

def multiprocessing_tqdm(function, array, cores=None, desc = None):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    # if front_num > 0:
    #     front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    # #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    # if n_jobs==1:
    #     return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as pool:
        #Pass the elements of array into function

        futures = [pool.submit(function, a) for a in array]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(concurrent.futures.as_completed(futures), **kwargs, desc = desc):
            pass
    out = []
    #Get the results from the futures. 
    for future in futures:
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out



def dict2csv(input: dict, filename: str, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}{filename}.csv", "w") as nf:
        for key in input.keys():
            nf.write(f"{key},{input[key]}\n")

def dict2json(input: dict, filename: str, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/{filename}.json", 'w',encoding='utf-8') as json_file:
        json.dump(input, json_file,indent=4, ensure_ascii=False)

def json2dict(filename, path):
    with open(f"{path}/{filename}.json") as json_file:
        data = json.load(json_file)
    return data

def str2txt(input: str, filename: str, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}{filename}.txt", "w") as nf:
        nf.write(input)

def list2txt(input: list, filename: str = "text_list", path = None):
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(f"{path}/{filename}.txt", "w") as nf:
        for element in input:
            nf.write(element+"\n")
            
def txt2list(filename, path):
    with open(f"{path}/{filename}.txt") as txt_file:
        txt_list = [line.strip() for line in txt_file]
    return txt_list

def subsequences(sequence, n: int):
    list_of_tuples = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    return list_of_tuples

def list2csv(input: list, filename: str, directory: str):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(f"/{directory}{filename}.csv", "w", newline="") as nf:
        wr = csv.writer(nf, quoting=csv.QUOTE_ALL)
        wr.writerows(input)

def csv2list(file: str, directory: str, n_start=None, n_end=None):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(f"{directory}{file}.csv", "r") as f:
        csv_reader = csv.reader(f)
        my_list = [
            (line[0], ast.literal_eval(line[1]), ast.literal_eval(line[2]))
            for line in take(n_end, csv_reader)
        ]
    return my_list[n_start:n_end]

# TODO: Really needed here?
def if_none_disable(x):
    if x == None:
        x = "disable"
    return x


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(itertools.islice(iterable, n))

def marginalize(joint_dist:dict,side="left"):
    print(f"Marginalizing to the {side}")
    marginal = Counter()
    for (l,r),v in joint_dist.items():
        marginal[r if side == "right" else l] += v
    return dict(marginal.most_common())

def normalize_dict(dictionnary: dict):
    norm = sum(dictionnary.values())
    prob_dict = {k: v / norm for k, v in dictionnary.items()}
    return prob_dict

def pmi(matrix, alpha=.75, type_pmi="sppmi"):
    """
    """
    # Taken from Kaggle, and modified

    if type_pmi == "nopmi":
        return matrix

    if type_pmi not in {"pmi","ppmi","spmi","sppmi","ssppmi","nopmi"}:
        pmi = "sppmi"
        print("Unknown type of PMI. Continuing with smoothed positive PMI (SPPMI)")
    num_skipgrams = matrix.sum()
    matrix_items = {(i, j): matrix[i,j] for i, j in zip(*matrix.nonzero())}
    # assert(sum(matrix_items.values()) == num_skipgrams)



    # for creating sparse matrices
    row_indxs = []
    col_indxs = []

    pmi_dat_values = []    # pointwise mutual information
    ppmi_dat_values = []   # positive pointwise mutual information
    spmi_dat_values = []   # smoothed pointwise mutual information
    sppmi_dat_values = []  # smoothed positive pointwise mutual information

    # reusable quantities

    sum_over_contexts = np.array(matrix.sum(axis=1)).flatten()
    sum_over_terms = np.array(matrix.sum(axis=0)).flatten()

    # smoothing
    sum_over_terms_alpha = sum_over_terms**alpha
    nca_denom = np.sum(sum_over_terms_alpha)

    for (tok_terms, tok_context), sg_count in matrix_items.items():
        # here we have the following correspondance with Levy, Goldberg, Dagan
        #========================================================================
        #   num_skipgrams = |D|
        #   nwc = sg_count = #(w,c)
        #   Pwc = nwc / num_skipgrams = #(w,c) / |D|
        #   nw = sum_over_cols[tok_ingredient]    = sum_over_contexts[tok_ingredient] = #(w)
        #   Pw = nw / num_skipgrams = #(w) / |D|
        #   nc = sum_over_rows[tok_context] = sum_over_ingredients[tok_context] = #(c)
        #   Pc = nc / num_skipgrams = #(c) / |D|
        #
        #   nca = sum_over_rows[tok_context]^alpha = sum_over_ingredients[tok_context]^alpha = #(c)^alpha
        #   nca_denom = sum_{tok_content}( sum_over_ingredients[tok_content]^alpha )

        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_terms]
        Pw = nw / num_skipgrams
        
        # note 
        # pmi = log {#(w,c) |D| / [#(w) #(c)]} 
        #     = log {nwc * num_skipgrams / [nw nc]}
        #     = log {P(w,c) / [P(w) P(c)]} 
        #     = log {Pwc / [Pw Pc]}

        if type_pmi == "pmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = np.log(Pwc/(Pw*Pc))   
        elif type_pmi == "ppmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = max(np.log(Pwc/(Pw*Pc)), 0)
        elif type_pmi == "spmi":
            nca = sum_over_terms_alpha[tok_context]
            Pca = nca / nca_denom
            pmi = np.log(Pwc/(Pw*Pca))
        elif type_pmi == "sppmi":
            nca = sum_over_terms_alpha[tok_context]
            Pca = nca / nca_denom  
            pmi = max(np.log(Pwc/(Pw*Pca)), 0)
        elif type_pmi == "ssppmi":
            nc = sum_over_terms[tok_context]
            Pc = nc / num_skipgrams
            pmi = max(np.log(Pwc/(Pw*Pc*alpha)), 0)
        
        row_indxs.append(tok_terms)
        col_indxs.append(tok_context)
        pmi_dat_values.append(pmi)
            
    pmi_mat = csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))

    print('Done')
    return pmi_mat

def parallel_processes(processes:list):
    """
    Warning: this produces global variables "parallel_process_n" which are deleted at the end of the computation.
    """
    st = time.perf_counter()
    np = len(processes)
    print(f"Computing {np} processes in parallel...")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=np)
    for n, process in zip(range(np),processes):
        variable_name = f"parallel_process_{n}"
        globals()[variable_name] = executor.submit(*process)
    for n in range(np):
        variable_name = f"parallel_process_{n}"
        globals()[variable_name] = eval(variable_name).result()
    fin = time.perf_counter()
    print(f"{np} processes computed in {round(fin-st,2)} secs.\n")
    result = [eval(f"parallel_process_{n}") for n in range(np)]
    for n in range(np):
        del globals()[f"parallel_process_{n}"]
    return result

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def delete_duplicates(x):
    return list(dict.fromkeys(x))

def clear_df(df):
    mask = df.applymap(lambda x: x is None)
    cols = df.columns[(mask).any()]
    for col in df[cols]:
        df.loc[mask[col], col] = ''
    return df

def df(list_of_list, keys = None):
    raw_df = pd.DataFrame(list_of_list,index=keys)
    return clear_df(raw_df.T)

# Draft functions

# def csv2list_raw(file: str, directory: str, n_start=None, n_end=None):
#     if not os.path.isdir(directory):
#         os.makedirs(directory)
#     with open(f"{directory}{file}.csv", "r") as f:
#         csv_reader = csv.reader(f)
#         my_list = [
#             [ast.literal_eval(item) for item in line]
#             for line in take(n_end, csv_reader)
#         ]
#     return my_list[n_start:n_end]

def csv2list_raw(file: str, directory: str, n_start=None, n_end=None,progress=False):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(f"{directory}{file}.csv", "r") as f:
        csv_reader = csv.reader(f)
        my_list = []
        if progress:
            csv_reader_it = tqdm(take(n_end, csv_reader),desc = f"Loading: {file}")
        else:
            csv_reader_it = take(n_end, csv_reader)
        for line in csv_reader_it:
            list_line = [ast.literal_eval(item) for item in line]
            my_list.append(list_line)
    return my_list[n_start:n_end]

def plot_scatter_line(
    x,
    y,
    trace_name = None,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    line_shape='spline',
    add_trace = None
    ):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name = trace_name, line_shape=line_shape, mode='lines+markers'))
    
    if add_trace != None:
        if isinstance(add_trace,tuple):
            add_trace = [add_trace]
        for trace in add_trace:
            fig.add_trace(go.Scatter(x=trace[0], y=trace[1], name = trace[2], line_shape=line_shape, mode='lines+markers')) # TODO: rewrite with **kwargs


    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=xaxis_title,
        
        )
    return fig

def scatter_plot(data:list,labels=None,legends=None,size=(15,10)):
    fig, ax = plt.subplots(figsize=size)  # Create a figure and an axes.
    for dataset in data:
        ax.scatter(range(len(dataset)) if labels==None else labels,dataset,label="Data")
    ax.set_xlabel('x')  # Add an x-label to the axes.
    ax.set_ylabel('f(x)')  # Add a y-label to the axes.
    ax.set_title("Title")  # Add a title to the axes.
    ax.legend(range(1,len(data)+1) if legends==None else legends)  # Add a legend.
    return ax

profiler = """
from pyinstrument import Profiler
import sys
def profiler(process):
    profile = Profiler()

    profile.start()
    
    try:
        exec(process)
    except:
        print(f"{sys.exc_info()}")

    profile.stop()

    return print(profile.output_text(unicode=True, color=True))"""


# value and index of max of matrix, in case needed
# np.array(np.unravel_index(np.argsort(voc_matrix.toarray().flatten(), axis=0)[-10:], voc_matrix.shape)).T[::-1]

####### TEST BUILD VOC, EFFACER #######

import regex as re
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from functools import partial, reduce
import operator
# import numpy as np
# from collections import Counter
# import socket
# socket_name = socket.gethostname()
# if any(name in socket_name for name in {"Gianni","vpn"}):
#     from tqdm.notebook import tqdm, trange
# else:
#     from tqdm.auto import tqdm, trange

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