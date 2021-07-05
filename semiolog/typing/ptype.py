import numpy as np
from scipy.signal import argrelmin, argrelextrema

from ..util import subsequences, plot_scatter_line

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
        self.len = parad_chain.len
        self.probs = parad_chain.probs
        self.labels = parad_chain.labels
        self.indexes = parad_chain.indexes
        self.types = [ptype for ptype in parad_chain.types]
        self.func_scores = np.array([ptype.func_score_s for ptype in parad_chain.types])
        self.scores_ext = list(argrelextrema(self.func_scores, np.greater)[0])
        self.scores_min = list(argrelmin(self.func_scores)[0])
        self.scores_max = list(argrelmin(self.func_scores)[0])

        self.scores_maxs_pair = subsequences([0]+self.scores_max+[len(self.func_scores)],2)

        self.phrases = []
        self.phrases_span = []
        for l,r in self.scores_maxs_pair:

            self.phrases.append({k:type for k,type in zip(parad_chain.labels[l:r],[True if s==min(self.func_scores[l:r]) else False for s in self.func_scores[l:r]])})

            self.phrases_span.append({l:s for l,s in zip(parad_chain.labels[l:r],parad_chain.spans[l:r])})

        self.phrases_nodes = {"".join(phr.keys()):(list((phr.values()))[0][0],list((phr.values()))[-1][-1]) for phr in self.phrases_span}

    def __getitem__(self, index:str):
        return self.types[index]
    
    def plot_scores(self, show_mean = False, show_probs = False):

        if show_probs == True:

            scaling_factor = np.max(self.probs)/np.max(self.func_scores)

        fig = plot_scatter_line(
            x=self.indexes,
            y=self.func_scores,
            trace_name= "Func. Scores",
            title=" ".join(self.labels),
            xaxis_title='Paradigm',
            yaxis_title='Functional Score',
            add_trace = (self.indexes, np.array(self.probs)/scaling_factor, "Probs.") if show_probs else None
            )

        if show_mean == True:
            score_mean = np.mean(self.func_scores)
            fig.add_shape(
                type="line",
                x0=0, x1 = self.len-1, y0 = score_mean, y1 = score_mean, 
                line=dict(
                    color="LightSeaGreen",
                    width=2,
                    dash="dash",
                ))
        


        return fig