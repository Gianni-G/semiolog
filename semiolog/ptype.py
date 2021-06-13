import numpy as np

class Type:

    def __init__(self,parad,semiotic) -> None:

        self.global_probs = np.array([semiotic.vocab.prob.get(k,0) for k in parad.keys_t])
        if parad.len_truncate == 0:
             self.mass = 0
             self.func_score = 1
        else:
            self.mass = np.mean(self.global_probs)
            self.func_score = self.mass/parad.len_truncate

        self.global_probs_s = np.array([semiotic.vocab.prob.get(k,0) for k in parad.keys_t_soft])
        if parad.len_truncate_soft == 0:
             self.mass_s = 0
             self.func_score_s = 1
        else:
            self.mass_s = np.mean(self.global_probs_s)
            self.func_score_s = self.mass_s/parad.len_truncate_soft



        # self.func = 

def chain_type(parad_chain, semiotic):

    types = [Type(parad,semiotic) for parad in parad_chain]

    return types