from scipy.stats import entropy
import numpy as np
import string

from ..util_g import df as slg_df

class Paradigm:
    def __init__(self,parad:dict,semiotic,cum_thres=.5) -> None:
        self.len = len(parad)
        self.keys = tuple(parad.keys())

        self.values = np.array(list(parad.values()))

        self.entropy = entropy(self.values)
        self.cumsum = np.cumsum(self.values)

        self.len_truncate = len([i for i in self.cumsum if i <= cum_thres])
        self.keys_t = tuple(list(self.keys)[:self.len_truncate])
        self.values_t = self.values[:self.len_truncate]

        parad_log = np.log(self.values)
        parad_soft_offset = parad_log+(-np.min(parad_log))
        self.soft_dist = parad_soft_offset/sum(parad_soft_offset)
        self.cumsum_soft = np.cumsum(self.soft_dist)

        self.len_truncate_soft = len([i for i in self.cumsum_soft if i <= cum_thres])
        self.keys_t_soft = tuple(list(self.keys)[:self.len_truncate_soft])
        self.values_t_soft = self.soft_dist[:self.len_truncate_soft]


    def __repr__(self) -> str:
        return str(self.keys)

class Paradigmatizer:
    
    def __init__(self) -> None:
        pass

    def __call__(self,chain):
        exclude_punctuation = chain.semiotic.config["paradigmatic"]["exclude_punctuation"]
        sent_mask = [" ".join([token.label for token in chain.mask(i)]) for i in range(chain.len)]
        parads = []
        for sent in sent_mask:
            parad = {i['token_str']:i['score'] for i in chain.semiotic.paradigmatic.unmasker(sent) if exclude_punctuation and i['token_str'] not in string.punctuation}
            parads.append(Paradigm(parad,chain.semiotic,chain.semiotic.config["paradigmatic"]["cumulative_sum_threshold"]))

        chain.paradigms = parads
        for token,parad in zip(chain,parads):
            token.paradigm = parad


class ParadigmChain:

    def __init__(self,chain) -> None:
        self.semiotic = chain.semiotic
        self.paradigms = [token.paradigm for token in chain.tokens]
        self.keys = [token.paradigm.keys for token in chain.tokens]
        self.keys_t = [token.paradigm.keys_t for token in chain.tokens]
        self.keys_t_soft = [token.paradigm.keys_t_soft for token in chain.tokens]

        self.labels = [token.label for token in chain.tokens]
        self.spans = [token.span for token in chain.tokens]
        self.indexes = [f"{token.label}_{token.position}" for token in chain.tokens]
    
    def __getitem__(self, index:str):
        return self.paradigms[index]
    
    def df(self,keys=None):
        if keys == "keys":
            return slg_df(self.keys, self.indexes)
        elif keys == "keys_t":
            return slg_df(self.keys_t, self.indexes)
        else:
            return slg_df(self.keys_t_soft, self.indexes)