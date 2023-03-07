
import numpy as np
import quimb.tensor as qtn
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

        self.M_t = qtn.Tensor(self.M, inds=[f"k{i}" for i in range(self.M.ndim)], tags = "M")

        self.built = True

    def partial_trace(self, index):

        if not self.built:
            return "SLG [E]: Tensor not built. Run the `build()` method on the object"
        if index >= len(self.M_t.inds):
            return "SLG [E]: The index is greater than the tensor rank"
        
        self.M_tH = self.M_t.H
        self.M_tH.drop_tags("M")
        self.M_tH.add_tag("Mh")

        self.M_tH.reindex({ind:(ind if i!= index else f"cut_{index}") for i,ind in enumerate(self.M_tH.inds)}, inplace=True)
        self.M_tn = self.M_t & self.M_tH
        self.M_pt = self.M_tn^...
        self.M_pt = self.M_pt.data

        self.plot_pt = plot_hm(
            z = self.M_pt,
            x = self.elements,
            y = self.elements)