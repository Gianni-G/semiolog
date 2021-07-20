import regex as re
from collections import Counter

def findall_contexts(chain,best_pair_string,re_voc_l,re_voc_r):
    contexts = re.findall(re_voc_l+best_pair_string+re_voc_r, chain, overlapped=True)
    return contexts


def find_best_pair(chain_list):
    pair_count = Counter()
    for pair in list(zip(chain_list, chain_list[1:])):
        pair_count[pair] += 1
    return pair_count