
import numpy as np
from scipy import sparse, linalg
from itertools import product, combinations
from collections import defaultdict
import jax.numpy as jnp
from .util import plot_hm, pmi, coolwarm
from .vocabulary import nGram
import jax.scipy as jsp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    __IPYTHON__
    from tqdm.notebook import tqdm, trange
except NameError:
    from tqdm.auto import tqdm, trange

class Tensor():

    def __init__(self,semiotic):
        self.semiotic = semiotic
        self.built = False

    def build(self, dims, rank, filter_thres = 10, exclude_punctuation = False, sparse = True):
        #TODO assert ngrams exist

        # Excluding punctuation has the consequence of exludes ngrams that contain at least one punctuation element. However, stats are not corrected. E.g. the 3-gram ('a','.','c') contains one punctuation mark, so it will be excluded, but its frequency should in principle be added to the bare freq of ('a','b'), which is not done here.

        if exclude_punctuation:
            self.elements = sorted([k for k in self.semiotic.vocab.alpha.keys() if k not in self.semiotic.vocab.punctuation][:dims])
        else:
            self.elements = sorted(list(self.semiotic.vocab.alpha.keys())[:dims])
        self.elements_dict = {k:i for i,k in enumerate(self.elements)}
        self.dims = len(self.elements)
        self.rank = rank
        self.shape = (self.dims,)*self.rank
        self.sparse = sparse

        # ng_atts = sorted([att for att in self.semiotic.vocab.__dict__.keys() if att.startswith("ng")])

        if self.rank == 1:
            ngs = nGram(from_dict=self.semiotic.vocab.alpha)
        else:
            ngs = getattr(self.semiotic.vocab,f"ng{rank}")

        if sparse:

            self.T_freq = defaultdict(int)
            for k,v in tqdm(ngs.freq.items()):
                if v>filter_thres:
                    if set(k).issubset(set(self.elements)):
                        indeces = tuple([self.elements_dict[i] for i in k])
                        self.T_freq[indeces] = v
            del ngs

        else:

            self.T_freq = np.zeros([dims]*rank)
            for k,v in ngs.freq.items():
                if v>filter_thres:
                    if set(k).issubset(set(self.elements)):
                        indeces = tuple([self.elements_dict[i] for i in k])
                        self.T_freq[indeces] = v
            del ngs

        self.built = True

    def partial_trace(
        self,
        terms,
        normalize = "matrix",
        center = False
        ):

        self.center = center

        if isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

        if not self.built:
            return "SLG [E]: Tensor not built. Run the `build()` method on the object"

        norm_set = {"cols","matrix"}
        if normalize != None and normalize not in norm_set:
            return f"SLG [E]: Unknown normalization argument ({normalize}). Possible arguments are: {norm_set}"

        if max(self.terms) >= self.rank:
            return "SLG [E]: At least one term index is greater than the tensor rank"
        
        #TODO: verify that term is completely well formed (list of integers, no repetition, not all ranks, etc.)

        self.contexts = [i for i in range(self.rank) if i not in set(self.terms)]

        blank_t = ["_"]*self.rank
        for i in self.terms:
            blank_t[i] = self.elements
        self.terms_labels = ["".join(t) for t in product(*blank_t)]

        # # Computing context labels can be very expensive and they're usually not useful (other than to plot small matrices)

        if self.dims <=50 and self.rank<=4:
            blank_c = ["¯"]*self.rank
            for i in self.contexts:
                blank_c[i] = self.elements
            self.context_labels = ["".join(t) for t in product(*blank_c)]

        if self.sparse:

            indices = np.array(list(self.T_freq.keys()))
            vals = np.array(list(self.T_freq.values()))

            cols = indices[:,self.terms] # Terms as columns, according to Anel&Gastaldi
            rows = indices[:,self.contexts] # Contexts as rows, according to Anel&Gastaldi

            # reshape indices
            if len(self.terms)>1:
                poly_cols = [self.dims**p for p in reversed(range(cols.shape[1]))]
                cols = (cols*poly_cols).sum(axis=1)
            else:
                cols = cols.flatten()

            if len(self.contexts)>1:
                poly_rows = [self.dims**p for p in reversed(range(rows.shape[1]))]
                rows = (rows*poly_rows).sum(axis=1)
            else:
                rows = rows.flatten()


            self.M = sparse.coo_array((vals, (rows, cols)), shape=(self.dims**(self.rank-len(self.terms)),self.dims**(self.rank-len(self.contexts))))
            self.M = self.M.tocsc()

            del indices, vals, cols, rows


            if normalize == "cols" :
                with np.errstate(divide='ignore'):
                    norm_coeff = 1/(self.M.sum(axis=0))
                self.M = self.M.multiply(norm_coeff)

            elif normalize == "matrix":
                with np.errstate(divide='ignore'):
                    norm_coeff = 1/(self.M.sum())
                self.M = self.M.multiply(norm_coeff)
            
            # We take sqrt to retrieve probabilities when contrating, following Bradley&Terilla
            self.M = self.M.sqrt()
            self.M = self.M.tocsc()

            cttc = (self.M.T)@self.M

            # center the matrix
            if self.center:
                # Warning: this centering works for the particular normalization by colums of the TC matrix (and CT the transpose of TC). Not intended for other forms of normalization
                mean = self.M.mean(axis=1)
                mtc = mean.T@self.M
                mct = mtc.reshape((mtc.size,1))
                mm = mean.T@mean

                #Degbug: erase after debugging
                self.cttc = cttc
                self.mtc = mtc
                self.mct = mct
                self.mm = mm
                #End of Debug

                self.pt = (cttc - mtc - mct) + mm
                self.pt = self.pt/(self.pt.shape[0]-1)
            else:
                self.pt = cttc.toarray()
            
            del cttc, mtc, mct, mm

        else:

            # Move axis and reshape
            self.M = np.moveaxis(self.T_freq, self.terms, list(range(len(self.terms))))
            self.M = self.M.reshape((self.dims**len(self.terms), self.dims**len(self.contexts)))
            self.M = self.M.T # Terms as columns, according to Anel&Gastaldi
        
            if normalize == "cols":

                col_sums = self.M.sum(axis=0)
                self.M = np.divide(self.M, col_sums, out=np.zeros_like(self.M), where=col_sums!=0)

                del col_sums

            elif normalize == "matrix":
                self.M = self.M/self.M.sum()

            # We take sqrt to retrieve probabilities when contrating, following Bradley&Terilla
            self.M = np.sqrt(self.M)

            # center the matrix
            if self.center:
                self.M = (self.M.T - self.M.mean(axis=1)).T

            self.pt = self.M.T@self.M

            if self.center:
                self.pt = self.pt/(self.pt.shape[0]-1)


    def pt_svd(self, top_w = None):

        # Compute full SVD
        if self.pt.ndim > 2:
            "SLG [E]: The partial trace is a tensor of rank > 2. SVD is not implemented for these cases"
        self.pt_U, self.pt_s, self.pt_Vh = jnp.linalg.svd(self.pt, 
                                    full_matrices=False, # It's not necessary to compute the full matrix of U or V
                                    compute_uv=True,
                                    )
        
        # TODO: Compute only first singular values (top_w)

    def pt_eig(self, top_w = 10):

        # Compute eigenvalues and eigenvectors
        if self.pt.ndim > 2:
            "SLG [E]: The partial trace is a tensor of rank > 2. Eigenvectors are not implemented for these cases"


        N = self.pt.shape[0]

        # Centering breaks the sparsity, hence we can't benefit from sparse algorithms here
        if self.center:
            self.eig_w, self.eig_v = jsp.linalg.eigh(self.pt)
        else:
            self.eig_w, self.eig_v = sparse.linalg.eigsh(self.pt, k=top_w, which='LM', v0=None)

        # order eigenvectors by greatest eigenvalue
        idx = self.eig_w.argsort()[::-1]   
        self.eig_w = np.array(self.eig_w[idx])
        self.eig_v = np.array(self.eig_v[:,idx])

    
    def plot(self, data, x=None, y=None):
        
        if type(data) in {np.ndarray,np.matrix}:
            z = data
        elif data == "pt":
            z = self.pt
            x = y = self.terms_labels
        elif data == "svd":
            z = jnp.diag(self.pt_s) @ self.pt_Vh
            x = self.terms_labels
            y = [f"d{i}" for i in range(self.dims)]
        elif data == "eig":
            z = (self.eig_v @ jnp.diag(self.eig_w/self.eig_w.sum())).T
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

    def plot_eig_hm(self, top_w = 10, top_d = 10, term = None):
        
        z = (self.eig_v @ np.diag(self.eig_w/self.eig_w.sum())).T

        if term == None:
            dbk = []
            dbv = []
            axis_n = min(top_w,z.shape[0])
            for d in range(axis_n):
                dbi = sorted(list(zip(z[d], self.terms_labels)))
                
                dbk_i_neg = [k for v,k in dbi[:top_d]]
                dbk_i_pos = [k for v,k in dbi[-top_d:]]
                
                dbv_i_neg = [v for v,k in dbi[:top_d]]
                dbv_i_pos = [v for v,k in dbi[-top_d:]]

                dbk.append(dbk_i_neg+["..."]+dbk_i_pos)
                dbv.append(dbv_i_neg+[0]+dbv_i_pos)

                vals = np.array(dbv).T
                labels = np.array(dbk).T
            
                fig = go.Figure(

                    data=go.Heatmap(dict(
                        z= vals,
                        x = [f"D {i+1}" for i in range(top_w)],
                        y = [str(-(top_d-i)) for i in range(top_d)] + ["0"] + [str(i+1) for i in range(top_d)],
                        colorscale = coolwarm,
                        zmid = 0,
                        xgap = 1,
                        ygap = 1,
                        text = labels,
                        texttemplate="%{text}",
                        textfont={"size":11,"family":"Courier New", "color": 'white',}
                        )
                        ))
        
        else:
            if len(self.terms)>1:
                return f"SLG [E]: Ploting terms is not yet implemented for partial traces of more than one index"
            if term not in self.elements:
                return f"SLG [E]: The term '{term}' is out of the vocabulary"
            term = self.elements_dict[term]
            eig_sort = np.argsort(z[:top_w])
            term_indeces = np.where(eig_sort==term)
            neighbour_indices = []
            for i,j in zip(*term_indeces):
                indices = eig_sort[i,max(0,j-top_d):j+top_d+1]
                if j-top_d<0:
                    indices = np.concatenate((np.array([-1]*(top_d-j)),indices))
                if j+top_d >= z.shape[1]:
                    indices = np.concatenate((indices,np.array([-1]*(j+top_d+1-z.shape[1]))))
                neighbour_indices.append(indices)
            neighbour_indices = np.array(neighbour_indices)
            vals = np.zeros(neighbour_indices.shape)
            for i in range(neighbour_indices.shape[0]):
                for j in range(neighbour_indices.shape[1]):
                    if neighbour_indices[i,j] != -1:
                        vals[i,j]=z[i,neighbour_indices[i,j]]
            labels = np.vectorize(lambda x: "" if x==-1 else self.terms_labels[x].upper() if x==term  else self.terms_labels[x])(neighbour_indices)

            vals = vals.T
            labels = labels.T

            fig = make_subplots(rows=2, cols=1,row_heights=[0.1, 0.9])
            
            term_v = z.T[term]
            dist = (z.T@term_v)
            dist_indices = list(reversed(list(np.argsort(dist).flatten())[-top_d:]))
            neighbors = np.array([[dist[i] for i in dist_indices]])
            neighbors = neighbors/(term_v@term_v)
            
            fig.append_trace(go.Heatmap(dict(
                    z= neighbors,

                    # x = [f"D {i+1}" for i in range(top_t)],
                    # y = [str(-(top_k-i)) for i in range(top_k)] + ["0"] + [str(i+1) for i in range(top_k)],
                    colorscale = coolwarm,
                    zmid = 0,
                    xgap = 1,
                    ygap = 1,
                    text = [[self.elements[i] if i!=term else self.elements[i].upper() for i in dist_indices]],
                    texttemplate="%{text}",
                    textfont={"size":11,"family":"Courier New", "color": 'white',},
                    )),
                row=1, col=1)

            fig.append_trace(go.Heatmap(dict(
                    z= vals,
                    x = [f"D {i+1}" for i in range(top_w)],
                    y = [str(-(top_d-i)) for i in range(top_d)] + ["0"] + [str(i+1) for i in range(top_d)],
                    colorscale = coolwarm,
                    zmid = 0,
                    xgap = 1,
                    ygap = 1,
                    text = labels,
                    texttemplate="%{text}",
                    textfont={"size":11,"family":"Courier New", "color": 'white',}
                    )),
                row=2, col=1)


        if term == None:
            fig.update_layout(
                yaxis = dict(
                    # scaleanchor = 'x',
                    autorange="reversed"
                    ),
                xaxis = dict(
                    side="top"
                    ),
                plot_bgcolor='rgba(0,0,0,0)',
                autosize=True,
                minreducedwidth=top_w*100,
                # width = top_t*100,
                height=top_d*80,
                font = dict(
                    family = "Courier New"
                ),
                )
        return fig
    

    def multiplot(self,x,y,z):

        def convert_to_color(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))*255
        fig = go.Figure()

        # Add all traces

        # align signs of eigenvectors across n_rank pt
        sign = np.sign(x[:,0]).reshape((x.shape[0],1))
        x = x * sign
        sign = np.sign(y[:,0]).reshape((y.shape[0],1))
        y = y * sign
        sign = np.sign(z[:,0]).reshape((z.shape[0],1))
        z = z * sign

        del sign

        x_color = convert_to_color(x)
        y_color = convert_to_color(y)
        z_color = convert_to_color(z)

        for i,(xi,yi,zi) in enumerate(zip(x,y,z)):

            fig.add_trace(
                go.Scatter3d(
                        x = xi, #if xi[0]>0 else -xi, # align signs of eigenvectors across n_rank pt
                        y = yi, #if yi[0]>0 else -yi,
                        z = zi, #if zi[0]>0 else -zi,
                        mode='markers+text',
                        text=self.elements,
                        textposition="bottom center",
                        marker = dict(
                            symbol = "circle",
                            color = [d for d in zip(
                                x_color[i],
                                y_color[i],
                                z_color[i],
                                )]
                        ),
                        visible= False if i>0 else True
                        )
                )

        del x_color, y_color, z_color

        # Add buttons
        n_traces = len(x)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.57,
                    y=1.2,
                    buttons = [

                        dict(
                            label=f"i = {i+1}",
                            # label=f"Context: {i+n_traces}",
                            method="update",
                            args=[{
                                "visible": [True if r==i else False for r in range(n_traces)]}]
                        )

                        for i in range(n_traces)
                    ]
                )
            ])

        max_val = np.abs(np.array([x,y,z])).max()
        max_val = 1
        fig.update_layout(
            scene = dict(
                aspectmode='cube',
                xaxis = dict(range=[-max_val,max_val],),
                yaxis = dict(range=[-max_val,max_val],),
                zaxis = dict(range=[-max_val,max_val],),
                ),
            margin=dict(l=0, r=0, b=0, t=0),
            )
        
        return fig
