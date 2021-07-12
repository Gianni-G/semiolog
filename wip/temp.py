from collections import Counter

# import socket
# if "Gianni" in socket.gethostname():
#     from tqdm.notebook import tqdm
# else:
from tqdm import tqdm

def parallel_chain(chain, n_of_parts):
    """
    Breaks the chain in n chunks to compute best pair of terms. Chunks are overlapping by one term, so as no pair of terms is lost due to the break.
    """
    if not isinstance(chain,list):
        chain = list(chain)
    chunk_size = int(len(chain) / n_of_parts)+1
    for i in range(0, len(chain), chunk_size):
        yield chain[i : i + chunk_size +1]
                
def find_best_pair(chain_list):

    pair_count = Counter()
    for pair in list(zip(chain_list, chain_list[1:])):
        pair_count[pair] += 1
    
    return pair_count