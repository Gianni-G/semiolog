import networkx as nx
import graphviz as gv

from . import util
from . import util_g
from .functive import Functive

class Tree:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = set(util_g.flatten(edges))
        self.root = [node for node in self.nodes if node not in [r for l, r in edges]][
            0
        ]
        self.leaves = {node for node in self.nodes if node not in [l for l, r in edges]}
        self.non_terminals = set(self.nodes) - self.leaves

    def graph(self):
        seg_tree = nx.DiGraph()
        seg_tree.add_edges_from(self.edges)

        return seg_tree

    def plot(self, red=None, grey=None, fname=f"seg_graph"):

        seg_tree = Tree.graph(self)

        seg_tree_graph = gv.Digraph(name=fname)

        tree_nodes_list = [(str(i), l) for l, i in list(seg_tree.nodes)]
        for node in tree_nodes_list:
            seg_tree_graph.node(
                *node, color="white", fontsize="30", fontname="garamond"
            )  # style="filled", color="grey")
        seg_tree_graph.attr("edge", color="slategrey")
        seg_tree_graph.edges([(str(p[1]), str(c[1])) for p, c in list(seg_tree.edges)])

        if grey == None:
            pass
        else:
            for node in [(str(i), l) for l, i in grey]:
                if node in tree_nodes_list:
                    seg_tree_graph.node(
                        *node,
                        style="dashed",
                        color="grey",
                        fontsize="30",
                        fontname="garamond",
                    )

        if red == None:
            pass
        else:
            for node in [(str(i), l) for l, i in red]:
                if node in tree_nodes_list:
                    seg_tree_graph.node(
                        *node, color="red", fontsize="30", fontname="garamond"
                    )

        return seg_tree_graph

def normalizer(sequence:str):
    norm = sequence.replace(" ", "")
    return norm

def pre_tokenize(sequence:str):

    return sequence



def tokenize(sequence: str, tokenizer):
    model = tokenizer[0]
    method = tokenizer[-1]
    if method == "sq":
        segments = " ".join(util.chain2seq(sequence, model.voc.freq)).split()

        spans = []
        for i in range(len(segments)):
            start_i = len("".join(segments[:i]))
            end_i = start_i + len(segments[i])
            spans.append((start_i, end_i))

        tokens = [Functive(segment,span,position,model) for position,(segment,span) in enumerate(zip(segments,spans))]

        tree_root = Functive(sequence, (0, len(sequence)), None, model)
        tree_root.children = tokens

        for token in tokens:
            token.head = tree_root

        return tokens

    # elif method == "tr":
    #     slg_edges = util.chain2tree(self.raw, self.model.voc.freq)
    #     return Tree(slg_edges)

    # elif method == "ud_original":
    #     doc = self.model.ud(self.raw)
    #     doc_idx = {
    #         token: (token.idx - i, token.idx - i + len(token))
    #         for i, token in enumerate(doc)
    #     }
    #     edges = [
    #         ((token.head.text, doc_idx[token.head]), (token.text, doc_idx[token]))
    #         for token in doc
    #         if token != token.head
    #     ]
    #     return Tree(edges)

    # elif method == "ud":

    #     doc = self.model.ud(self.raw)

    #     doc_idx = {token:(token.idx-i,token.idx-i+len(token)) for i,token in enumerate(doc)}

    #     ud_segs = []
    #     for token_doc in doc:
    #         subtrees = []
    #         # subtrees_control = []
    #         for token in token_doc.subtree:
    #             # subtrees_control.append(doc_idx[token])
    #             for i in doc_idx[token]:
    #                 subtrees.append(i)
    #         contiguous_breaks = [l for l,r in util_g.subsequences(subtrees,2) if l==r]
    #         real_breaks = [b for b in subtrees if b not in contiguous_breaks]
    #         real_intervals = sorted([(real_breaks[2*i],real_breaks[2*i+1]) for i in range(int(len(real_breaks)/2))])
    #         final_subtrees = [(self.norm[l:r],(l,r)) for l,r in real_intervals]
    #         for subtree_f in final_subtrees:
    #             ud_segs.append((token_doc,subtree_f))

    #     ud_segs_dict = dict(ud_segs)

    #     ud_tree = [(ud_segs_dict[token.head],ud_segs_dict[token]) for token in doc if ud_segs_dict[token]!= ud_segs_dict[token.head]]
    #     head_edges = [(node,(token.text,doc_idx[token])) for token,node in ud_segs if token.text!=node[0]]
    #     ud_tree+=head_edges

    #     edges = sorted(ud_tree, key=lambda x: (x[0][1][0],x[1][1][0])) # trick to order first on head and then on children

    #     return Tree(edges)

    # elif method == "cp":
    #     cp_sent = sorted(list(self.model.cp(self.raw).sents), key=len)[
    #         -1
    #     ]  # In case wrong analysis it picks the longest (sub)sentences recognized. This would need better treatment.
    #     test_label = str(cp_sent).replace(" ", "")
    #     init = [cp_sent]
    #     init_intervals = [(0, len(test_label))]
    #     edges = []
    #     it = 0
    #     while init != []:
    #         it += 1
    #         init_collect = []
    #         interval_collect = []
    #         for i, interval in zip(init, init_intervals):
    #             ch_label = [str(ch).replace(" ", "") for ch in i._.children]
    #             ch_int = []
    #             for n in range(1, len(ch_label) + 1):
    #                 label_int = (
    #                     interval[0] + len("".join(list(ch_label[: n - 1])))
    #                 ), interval[0] + len("".join(list(ch_label[:n])))
    #                 ch_int.append(label_int)
    #             node_list = [
    #                 (
    #                     (str(i).replace(" ", ""), interval),
    #                     (str(ch).replace(" ", ""), ch_i),
    #                 )
    #                 for ch, ch_i in zip(i._.children, ch_int)
    #             ]
    #             edges.append(node_list)
    #             init_collect += i._.children
    #             interval_collect += ch_int
    #         init = init_collect
    #         init_intervals = interval_collect

    #     edges = util_g.flatten(edges)
    #     return Tree(edges)

    # else:
    #     return print(
    #         "No recognizable model name. Try any of these: sq, tr, ud, cp."
    #     )


class Chain:
    def __init__(self, input_chain: str, model, method = "sq"):  # ud_model, cp_model):
        
        self.input = input_chain
        self.split = input_chain.split()
        self.norm = normalizer(input_chain)
        self.pre_tokens = pre_tokenize(self.norm)

        self.tokenizer = (model, method)

        self.tokens = tokenize(self.pre_tokens, self.tokenizer)

        self.labels = [token.label for token in self.tokens]
        self.segmented = " ".join(self.labels)

        # self.nodes_list = []
        # for i in range(len(self.split)):
        #     start_i = len("".join(self.split[:i]))
        #     end_i = start_i + len(self.split[i])
        #     self.nodes_list.append((self.split[i], (start_i, end_i)))
            
        # self.nodes = set(self.nodes_list)
        

    def __repr__(self) -> str:
        return f"Chain({self.input})"

    def __str__(self) -> str:
        return self.raw

    
