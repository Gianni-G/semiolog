
import numpy as np
import jax.numpy as jnp
from .util import plot_hm

class Tensor():

    def __init__(self,semiotic):
        self.semiotic = semiotic
        self.built = False

    def build(self, dims, rank, filter_thres = 10):
        #TODO assert ngrams exist
        self.elements = sorted(list(self.semiotic.vocab.alpha.keys())[:dims])
        self.dims = len(self.elements)
        self.elements_dict = {k:i for i,k in enumerate(self.elements)}

        ng_atts = sorted([att for att in self.semiotic.vocab.__dict__.keys() if att.startswith("ng")])

        ngs = getattr(self.semiotic.vocab,ng_atts[rank-1])

        M_raw = np.zeros([dims]*rank)
        for k,v in ngs.freq.items():
            if v>filter_thres:
                if set(k).issubset(set(self.elements)):
                    indeces = tuple([self.elements_dict[i] for i in k])
                    M_raw[indeces] = v
        M_norm = M_raw/M_raw.sum()
            
        self.M = np.sqrt(M_norm)
        # self.M = self.M - self.M.mean()
        # self.M = M_norm

        self.built = True

    def partial_trace(self, terms):

        if isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

        if not self.built:
            return "SLG [E]: Tensor not built. Run the `build()` method on the object"

        if max(self.terms) >= self.M.ndim:
            return "SLG [E]: At least one term index is greater than the tensor rank"
        
        self.contexts = [i for i in range(self.M.ndim) if i not in set(self.terms)]

        self.M_pt = np.tensordot(self.M, self.M.T, axes=(self.contexts,self.contexts[::-1]))

        self.plot_pt = plot_hm(
            z = self.M_pt,
            x = self.elements,
            y = self.elements)

    def pt_svd(self, truncate = None):
        # Taken from Zichen Wang (https://towardsdatascience.com/turbocharging-svd-with-jax-749ae12f93af)

        # Compute full SVD
        if self.M_pt.ndim > 2:
            "SLG [E]: The partial trace is a tensor of rank > 2. SVD is not implemented for these cases"
        self.pt_U, self.pt_s, self.pt_Vh = jnp.linalg.svd(self.M_pt, 
                                    full_matrices=False, # It's not necessary to compute the full matrix of U or V
                                    compute_uv=True,
                                    )
        self.plot_pt_svd = plot_hm(
            z = jnp.diag(self.pt_s) @ self.pt_Vh,
            x = self.elements,
            y = [f"t_{i}" for i in range(self.dims)])

