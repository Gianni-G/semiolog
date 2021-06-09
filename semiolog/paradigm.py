from scipy.stats import entropy
import numpy as np
from statistics import mean, geometric_mean

class Parad:
    def __init__(self,parad:dict,semiotic,cum_thres=.5) -> None:
        self.len = len(parad)
        self.keys = tuple(parad.keys())


        self.values = np.array(list(parad.values()))


        self.entropy = entropy(self.values)
        self.cumsum = np.cumsum(self.values)

        len_truncate = len([i for i in self.cumsum if i <= cum_thres])
        self.keys_t = tuple(list(self.keys)[:len_truncate])
        self.values_t = self.values[:len_truncate]

        parad_log = np.log(self.values)
        +(-np.min(self.values))
        parad_soft_offset = parad_log+(-np.min(parad_log))
        self.soft_dist = parad_soft_offset/sum(parad_soft_offset)
        self.cumsum_soft = np.cumsum(self.soft_dist)

        len_truncate_soft = len([i for i in self.cumsum_soft if i <= cum_thres])
        self.keys_t_soft = tuple(list(self.keys)[:len_truncate_soft])
        self.values_t_soft = self.soft_dist[:len_truncate_soft]

        self.global_probs = np.array([semiotic.voc.prob.get(k,0) for k in self.keys_t])
        if len_truncate == 0:
             self.mass = 0
             self.func_score = 1
        else:
            self.mass = np.mean(self.global_probs)
            self.func_score = self.mass/len_truncate

        self.global_probs_s = np.array([semiotic.voc.prob.get(k,0) for k in self.keys_t_soft])
        if len_truncate_soft == 0:
             self.mass_s = 0
             self.func_score_s = 1
        else:
            self.mass_s = np.mean(self.global_probs_s)
            self.func_score_s = self.mass_s/len_truncate_soft

    def __repr__(self) -> str:
        return str(self.keys)

def chain_paradigm(chain, semiotic, thres=0):

    sent_list = chain.split
    sent_mask = [" ".join([token if n!=i else "[MASK]" for n,token in enumerate(sent_list)]) for i in range(len(sent_list))]

    punctuation = {
        ".",":",",","…","'","’","′",'"',"•",";","`","-","“","...","?","!","/","&","–"
        }

    parads = []
    for sent in sent_mask:
        parad = {i['token_str']:i['score'] for i in semiotic.unmasker(sent) if i['score']>thres and i['token_str'] not in punctuation }

        parads.append(Parad(parad,semiotic))

    return parads