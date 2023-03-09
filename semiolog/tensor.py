
import numpy as np
from itertools import product, combinations
import jax.numpy as jnp
from .util import plot_hm, pmi

class Tensor():

    def __init__(self,semiotic):
        self.semiotic = semiotic
        self.built = False

    def build(self, dims, rank, filter_thres = 10):
        #TODO assert ngrams exist
        self.elements = sorted(list(self.semiotic.vocab.alpha.keys())[:dims])
        self.elements_dict = {k:i for i,k in enumerate(self.elements)}
        self.dims = len(self.elements)
        self.rank = rank

        ng_atts = sorted([att for att in self.semiotic.vocab.__dict__.keys() if att.startswith("ng")])

        ngs = getattr(self.semiotic.vocab,ng_atts[rank-1])

        self.M_freq = np.zeros([dims]*rank)
        for k,v in ngs.freq.items():
            if v>filter_thres:
                if set(k).issubset(set(self.elements)):
                    indeces = tuple([self.elements_dict[i] for i in k])
                    self.M_freq[indeces] = v

        self.M_norm = self.M_freq/self.M_freq.sum()
            
        # # Attempt to compute Pointwise Interaction Information and Pointwise Total Correlation
        
        # freqs = []
        # norms = []
        # for N in range(1,self.rank):
        #     freq = [self.M_freq.sum(axis=i) for i in combinations(range(self.rank), N)]
        #     norm = [f/f.sum() for f in freq]
        #     freqs.append(freq)
        #     norms.append(norm)
        
        # sums = freqs
        # M_pii = np.zeros(self.M_freq.shape)
        # M_ptc = np.zeros(self.M_freq.shape)
        # for i,j,k in product(range(self.dims), repeat=self.rank):

        #     ptc_denominator = sums[1][0][k]*sums[1][1][j]*sums[1][2][i]
        #     pii_denominator = self.M_freq[i,j,k]*ptc_denominator

        #     if ptc_denominator != 0:
        #         ptc_coeff = self.M_freq[i,j,k]/ptc_denominator
        #     else:
        #         ptc_coeff = 0
            
        #     if pii_denominator != 0:
        #         pii_coeff = (sums[0][0][j,k]*sums[0][1][i,k]*sums[0][2][i,j])/pii_denominator
        #     else:
        #         pii_coeff = 0

        #     M_pii[i,j,k] = np.log(pii_coeff) if pii_coeff>0 else 0
        #     M_ptc[i,j,k] = np.log(ptc_coeff) if ptc_coeff>0 else 0

        self.M = np.sqrt(self.M_norm)
        # self.M = M_pii

        self.built = True

    def partial_trace(self, terms, normalize = None, center = False):

        if isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

        if not self.built:
            return "SLG [E]: Tensor not built. Run the `build()` method on the object"

        if max(self.terms) >= self.M.ndim:
            return "SLG [E]: At least one term index is greater than the tensor rank"
        
        self.contexts = [i for i in range(self.M.ndim) if i not in set(self.terms)]

        blank_t = ["_"]*self.rank
        for i in self.terms:
            blank_t[i] = self.elements
        self.terms_labels = ["".join(t) for t in product(*blank_t)]

        blank_c = ["¯"]*self.rank
        for i in self.contexts:
            blank_c[i] = self.elements
        self.context_labels = ["".join(t) for t in product(*blank_c)]

        if normalize != None:
            M = np.moveaxis(self.M, self.terms, list(range(len(self.terms))))
            M = M.reshape((self.dims**len(self.terms), self.dims**len(self.contexts)))
            self.tc = M.T # Terms as columns, according to Anel&Gastaldi

        if normalize == "pmi":

            self.tc_pmi = pmi(self.tc, normalize = True)
            self.ct_pmi = self.tc_pmi.T

            if center:
                self.tc_pmi -= self.tc_pmi.mean(axis=0)
                self.ct_pmi -= self.ct_pmi.mean(axis=0)

            self.pt = self.ct_pmi@self.tc_pmi

            if center:
                self.pt = self.pt/(self.pt.shape[0]-1)
        
        elif normalize == "probs":
            
            # self.tc_probs = self.tc / self.tc.sum(axis=0)
            # self.ct_probs = self.tc.T / self.tc.T.sum(axis=0)

            col_sums = self.tc.sum(axis=0)
            self.tc_probs = np.divide(self.tc, col_sums, out=np.zeros_like(self.tc), where=col_sums!=0)

            row_sums = self.tc.T.sum(axis=0)
            self.ct_probs = np.divide(self.tc.T, row_sums, out=np.zeros_like(self.tc.T), where=row_sums!=0)

            # center the matrix
            if center:
                self.tc_probs -= self.tc_probs.mean(axis=0)
                self.ct_probs -= self.ct_probs.mean(axis=0)

            self.pt = self.ct_probs@self.tc_probs

            if center:
                self.pt = self.pt/(self.pt.shape[0]-1)

        else:
            self.pt = np.tensordot(self.M, self.M.T, axes=(self.contexts,self.contexts[::-1]))


    def pt_svd(self, truncate = None):

        # Compute full SVD
        if self.pt.ndim > 2:
            "SLG [E]: The partial trace is a tensor of rank > 2. SVD is not implemented for these cases"
        self.pt_U, self.pt_s, self.pt_Vh = jnp.linalg.svd(self.pt, 
                                    full_matrices=False, # It's not necessary to compute the full matrix of U or V
                                    compute_uv=True,
                                    )

    def pt_eig(self, truncate = None):

        # Compute eigenvalues and eigenvectors
        if self.pt.ndim > 2:
            "SLG [E]: The partial trace is a tensor of rank > 2. Eigenvectors are not implemented for these cases"

        self.eig_w, self.eig_v = np.linalg.eig(self.pt)

        # order eigenvectors by greatest eigenvalue
        idx = self.eig_w.argsort()[::-1]   
        self.eig_w = self.eig_w[idx]
        self.eig_v = self.eig_v[:,idx]

    
    def plot(self, data, x=None, y=None):
        
        if type(data) == np.ndarray:
            z = data
        elif data == "pt":
            z = self.pt
            x = y = self.terms_labels
        elif data == "svd":
            z = jnp.diag(self.pt_s) @ self.pt_Vh
            x = self.terms_labels
            y = [f"d{i}" for i in range(self.dims)]
        elif data == "eig":
            z = (self.eig_v @ jnp.diag(self.eig_w)).T
            x = self.terms_labels
            y = [f"d{i}" for i in range(self.dims)]
        else:
            return f"SLG [E]: Type of Data ({type(data)}) not recognized."
            
        if z.ndim>2:
            return "SLG [W]: Plot of tensors of rank>2 not implemented."
        else:
            fig = plot_hm(
                z = z,
                x = x,
                y = y)

            return fig
