import regex as re
from collections import Counter

def findall_contexts(chain,best_pair_string,re_voc_l,re_voc_r):
    contexts = re.findall(re_voc_l+best_pair_string+re_voc_r, chain, overlapped=True)
    return contexts

def findall_contexts_list(chain_list,best_pair,encode):
    
    merge_context_count_l = Counter()
    merge_context_count_r = Counter()
    pair_pair_count = 0
    pair_pair_overlap_control = [-4]

    for j,(a,b,c,d) in enumerate(zip(*[chain_list[i:] for i in range(4)])):
        if (b,c) == best_pair:
            merge_context_count_l[encode[a]] += 1
            merge_context_count_r[encode[d]] += 1
            
        # compute #(l,r)-(l,r)
        if (a,b,c,d) == best_pair*2 and j-pair_pair_overlap_control[-1]>3:
            pair_pair_count += 1
            pair_pair_overlap_control.append(j)
    
    return merge_context_count_l, merge_context_count_r, pair_pair_count


def find_best_pair(chain_list):
    pair_count = Counter()
    for pair in list(zip(chain_list, chain_list[1:])):
        pair_count[pair] += 1
    return pair_count

def agglutinate_list(chain_list,best_pair):
    best_pair_string = "".join(best_pair)
    index = [-2]
    for i,pair in enumerate(zip(chain_list,chain_list[1:])):
        if pair == best_pair and i!=index[-1]+1:
            index.append(i)
            chain_list[i] = best_pair_string
            chain_list[i+1] = "--"

    chain_list = list(filter("--".__ne__, chain_list))

    return chain_list