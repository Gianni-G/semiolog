from scipy.stats import entropy
import numpy as np

class Parad:
    def __init__(self,parad:dict,cum_thres=.7) -> None:
        self.len = len(parad)
        self.keys = tuple(parad.keys())
        self.values = np.array(list(parad.values()))
        self.mass = np.sum(self.values)
        self.entropy = entropy(self.values)
        self.cumsum = np.cumsum(self.values)
        len_truncate = len([i for i in self.cumsum if i <= cum_thres])
        self.keys_t = tuple(list(self.keys)[:len_truncate])
        self.values_t = self.values[:len_truncate]
    
    def __repr__(self) -> str:
        return str(self.keys)

def chain_paradigm(chain, unmasker, thres=0):

    sent_list = chain.split
    sent_mask = [" ".join([token if n!=i else "[MASK]" for n,token in enumerate(sent_list)]) for i in range(len(sent_list))]

    punctuation = {
        ".",":",",","…","'","’","′",'"',"•",";","`","-","“","...","?","!","/","&","–"
        }

    parads = []
    for sent in sent_mask:
        parad = {i['token_str']:i['score'] for i in unmasker(sent) if i['score']>thres and i['token_str'] not in punctuation }

        parads.append(Parad(parad))

    return parads