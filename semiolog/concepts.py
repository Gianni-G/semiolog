from tqdm.notebook import tqdm
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from itertools import product
from scipy import sparse
import graphviz as gv
import json

try:
    __IPYTHON__
    from tqdm.notebook import tqdm, trange
except NameError:
    from tqdm.auto import tqdm, trange

class Concepts():

    def __init__(self,semiotic):
        self.semiotic = semiotic
        self.tensor = semiotic.tensor
        self.built = False

    def build_matrix(
        self,
        cutoff = 0,
        sqrt = False,
        center = False,
        ):
        """
        If sqrt = True, then the cutoff gets automatically squared
        """

        self.cutoff = cutoff

        try:
            M = self.tensor.M.T.copy() # Transposing matrix here, to have terms ("objects") as rows and contexts ("features") as columns
        except:
            return "SLG [E]: The tensor's term-context matrix is not built. Please run build_tc_matrix() on the tensor first"
        
        if sqrt:
            if self.tensor.sparse:
                M = M.sqrt()
            else:
                M = np.sqrt(M)

            self.cutoff = np.sqrt(self.cutoff)

        if center:
            M_mean = M.mean(axis=0).reshape((1,M.shape[1]))
            M = M - M_mean

        self.M_bool = sparse.csr_matrix((M>self.cutoff))

        try:
            self.context_labels = self.tensor.context_labels
        except:
            self.context_labels = ["_".join(p) for p in product(self.tensor.elements,repeat=len(self.tensor.contexts))]

            blank_c = ["_"]*self.tensor.rank
            for i in self.tensor.contexts:
                blank_c[i] = self.tensor.elements
            self.context_labels = [" ".join(t) for t in product(*blank_c)]
        if len(self.tensor.terms) == 1:
            self.terms_labels = self.tensor.elements
        else:
            self.terms_labels = self.tensor.terms_labels

    def plot_M(
        self,
        ):

        return self.tensor.plot(self.M_bool.toarray()*1, self.context_labels, self.terms_labels)

    def build_concepts(
        self,
        ):

        c_terms = []
        c_contexts_atomic = {frozenset(set(self.M_bool.indices[i:j])) for i,j in zip(self.M_bool.indptr,self.M_bool.indptr[1:])}
        current = c_contexts_atomic
        diff = c_contexts_atomic
        for i in range(2,len(c_contexts_atomic)):
            # prod = tqdmprod(c_contexts_atomic, diff, leave=True)
            new = {frozenset(frozenset.intersection(*t)) for t in tqdm(product(c_contexts_atomic, diff),total=len(c_contexts_atomic)*len(diff))}
            diff = new - current
            if diff == set():
                print(f"No non-empty sets after i={i}")
                break
            current = current.union(diff)

        current = current.union({frozenset(range(self.M_bool.shape[1]))})
        current = current.union({frozenset()})
        c_contexts = sorted([tuple(c) for c in current],key = lambda x: len(x))

        for c_context in c_contexts:
            b = np.all(self.M_bool[:,c_context].toarray().astype(bool),axis=1)
            b = sparse.csr_array(b)
            # c_terms.append(b.indices[i:j] for i,j in zip(b.indptr,b.indptr[1:]))
            c_terms.append(tuple(b.indices))
        
        self.ids = list(zip(c_terms,c_contexts))
        self.ids_len = len(list(zip(c_terms,c_contexts)))

    def graph(
        self,
        thres = 0,
        thres_prod = True,
        isolated_nodes = True,
        reverse = False,
        labels = "tc",
        specs = True,
        score = True,
        plot = True,
        ):
        """
        labels: {"tc","t","c"}
        """

        step = 0
        # pbar = tqdm(desc = f"Step {step}/11", total = 11)
        pbar = tqdm(desc = "Graph", total = 12)

        self.G = nx.DiGraph()

        if not thres_prod: # This is a quick draft, can be simplified
            if isinstance(thres,int):
                thres = (thres,thres)
            
            if not ((isinstance(thres,tuple) or isinstance(thres,list)) and len(thres)==2):
                return f"SLG [E]: Incorrect 'thres' value: {thres}. When 'thres_prod' is False, 'thres' should be an integer or a (list or tuple) pair."

            thres_t, thres_c = thres

            nodes_filtered = []
            for i in self.ids:
                if all((len(i[0])>=thres_t,len(i[1])>=thres_c)):
                    nodes_filtered.append(i)
        
        else:
            if not isinstance(thres,int):
                return "SLG [E]: Incorrect 'thres' value: {thres}. When 'thres_prod' is False, 'thres' should be an integer."
            
            nodes_filtered = []
            for i in self.ids:
                if len(i[0])*len(i[1])>=thres:
                    nodes_filtered.append(i)

        self.thres = thres


        if len(nodes_filtered) == 0:
            return "SLG [I]: There are no formal concepts in this context."

        step += 1
        pbar.update(step)

        if reverse:
            for i,j in product(nodes_filtered,repeat=2):
                if i!=j:
                    if set(i[0]).issubset(j[0]):
                        self.G.add_edge(i,j)
                    elif set(j[0]).issubset(i[0]):
                        self.G.add_edge(j,i)
        else:
            for i,j in product(nodes_filtered,repeat=2):
                if i!=j:
                    if set(i[0]).issubset(j[0]):
                        self.G.add_edge(j,i)
                    elif set(j[0]).issubset(i[0]):
                        self.G.add_edge(i,j)

        step += 1
        pbar.update(step)

        if isolated_nodes:
            for i in nodes_filtered:
                self.G.add_node(i)
        else:
            if len(self.G.nodes()) == 0:
                return "SLG [I]: There are no no-isolated nodes in this graph."

        step += 1
        pbar.update(step)

        self.G = nx.transitive_closure_dag(self.G)
        self.G_len = len(self.G.nodes())

        step += 1
        pbar.update(step)



        # labels_t = [
        #     "{"
        #     +",".join([self.terms_labels[k]+","+("<br>" if p for p,k in enumerate(i)])
        #     +"}"
        #     +"<br>"

        #     for i,j in self.G.nodes]
        
        # labels_c = [
        #     "<br>"
        #     +"{"
        #     +",".join([self.context_labels[k] if (b)%3 != 0 or b==0 else "<br>"+self.context_labels[k] for b,k in enumerate(j[:6])])
        #     + (",…" if len(j)>6 else "")
        #     +"}"

        #     for i,j in self.G.nodes]

        labels_t = []
        labels_c = []
        for i,j in self.G.nodes:
            t = sorted([self.terms_labels[k]+"," for k in i[:12]])
            for l in range(1,int((len(t)-2)/4)+1):
                t.insert((l*5)-1,"<br>")
            t = "{"+("".join(t))[:-1]+("" if len(i)<=12 else ",…")+"}<br>"
            labels_t.append(t)

            c = sorted([self.context_labels[k]+"," for k in j[:6]])
            for l in range(1,int((len(j[:6])-2)/3)+1):
                c.insert((l*4)-1,"<br>")
            c = "<br>{"+("".join(c))[:-1]+("" if len(j)<=6 else ",…")+"}"
            labels_c.append(c)

        step += 1
        pbar.update(step)

        node_score = [len(node[0])*len(node[1]) for node in self.G.nodes]
        
        node_hover = [(
            labels_t[i]
            +"<br>"
            +"{"
            +",".join([self.context_labels[k] if (b)%3 != 0 or b==0 else "<br>"+self.context_labels[k] for b,k in enumerate(j)])
            +"}"
            +"<br><br>"
            +f"Score: {node_score[i]}"
            ) for i,(k,j) in enumerate(self.G.nodes)]

        min_score = min(node_score)
        max_score = max(node_score)

        step += 1
        pbar.update(step)

        self.Gv = gv.Digraph(name="Concepts", directory="./", strict=True)

        self.Gv.attr("node", fontname = "Courier New")
        self.Gv.attr("edge", color="grey80")

        self.Gv.edges(
            [(str(i), str(j)) for i, j in self.G.edges()]
        )

        if "c" in labels and "t" not in labels:
            node_text = labels_c
        else:
            node_text = labels_t

        for node,label,scor in zip(self.G.nodes,node_text,node_score):
            self.Gv.node(
                str(node),
                label.replace("<br>",""),
                style="filled",
                colorscheme = "reds9",
                color = ("5" if node_score == [] or max_score-min_score == 0 or score == False else str(int(((scor-min_score)/(max_score-min_score))*7+1))),
                # color="4",
                shape="oval",
            )

        step += 1
        pbar.update(step)

        json_string = self.Gv.pipe('json').decode()
        json_dict = json.loads(json_string)
        node_pos = {}
        for obj in json_dict['objects']:
            node_pos[eval(obj['name'])] = eval(obj['pos'])

        step += 1
        pbar.update(step)

        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = node_pos[edge[0]]
            x1, y1 = node_pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=.25, color='#888'),
            hoverinfo='none',
            mode='lines',
            )

        step += 1
        pbar.update(step)

        node_x = []
        node_y = []
        for node in self.G.nodes():
            x, y = node_pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            customdata=node_hover,
            hovertemplate="%{customdata}",
            marker=dict(
                showscale=score,
                colorscale="OrRd",
                color= node_score if score else None,
                size=20,
                colorbar= dict(
                    thickness=15,
                    xanchor='left',

                    
                    # title=,
                    # titleside='right'
                ),
                line_width=1))

        step += 1
        pbar.update(step)

        label_t_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='text',
            text=labels_t,
            textposition="top center",
            )

        label_c_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='text',
            text=labels_c,
            textposition="bottom center",
            )
        
        step += 1
        pbar.update(step)
        
        self.fig = go.Figure(
            data=[edge_trace, node_trace] + ([label_t_trace] if "t" in labels else []) + ([label_c_trace] if "c" in labels else []),
            layout=go.Layout(
                # title=f"Corpus: {self.semiotic.name}, Norm: {self.tensor.norm}, Cutoff: {self.cutoff}, Threshold: {self.thres}",
                # titlefont_size=36,
                font={"size":14,"family":"Courier New",},
                showlegend=False,
                hovermode='closest',
                margin=dict(b=30,l=5,r=5,t=40),
                annotations=[] if not specs else [ dict(
                    text=f"Corpus: {self.semiotic.name}, Norm: {self.tensor.norm}, Cutoff: {self.cutoff}, Threshold: {self.thres}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.03 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(node_y)*-0.7,max(node_y)*1.2]),
                ))

        step += 1
        pbar.update(step)

        pbar.close()
        if plot:
            return self.fig