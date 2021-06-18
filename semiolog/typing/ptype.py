import numpy as np

class PType:

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


class Typer:
    def __init__(self) -> None:
        pass

    def __call__(self,parad_chain):
        types = []
        for parad in parad_chain:
            types.append(PType(parad,parad_chain.semiotic))
        parad_chain.types = types

class TypeChain:

    def __init__(self,parad_chain) -> None:
        self.semiotic = parad_chain.semiotic
        self.types = [ptype for ptype in parad_chain.types]

    def __getitem__(self, index:str):
        return self.types[index]
