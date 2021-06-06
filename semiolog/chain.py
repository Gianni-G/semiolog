import networkx as nx
import graphviz as gv

from . import util
from . import util_g

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


class Chain:
    def __init__(self, raw_chain: str, model):  # ud_model, cp_model):
        self.raw = raw_chain
        self.norm = raw_chain.replace(" ", "")
        self.split = raw_chain.split()
        self.nodes = []
        for i in range(len(self.split)):
            start_i = len("".join(self.split[:i]))
            end_i = start_i + len(self.split[i])
            self.nodes.append((self.split[i], (start_i, end_i)))
        self.nodes = set(self.nodes)
        self.model = model

    def __repr__(self) -> str:
        return f"Chain({self.raw})"

    def __str__(self) -> str:
        return self.raw

    def segment(self, model_name):

        if model_name == "sq":
            segments = " ".join(util.chain2seq(self.raw, self.model.voc.freq)).split()

            intervals = []
            for i in range(len(segments)):
                start_i = len("".join(segments[:i]))
                end_i = start_i + len(segments[i])
                intervals.append((start_i, end_i))

            slg_root = (self.norm, (0, len(self.norm)))

            slg_edges = [(slg_root, node) for node in zip(segments, intervals)]

            return Tree(slg_edges)

        elif model_name == "tr":
            slg_edges = util.chain2tree(self.raw, self.model.voc.freq)
            return Tree(slg_edges)

        elif model_name == "ud":
            doc = self.model.ud(self.raw)
            doc_idx = {
                token: (token.idx - i, token.idx - i + len(token))
                for i, token in enumerate(doc)
            }
            ud_tree = [
                ((token.head.text, doc_idx[token.head]), (token.text, doc_idx[token]))
                for token in doc
                if token != token.head
            ]
            return Tree(ud_tree)

        elif model_name == "cp":
            cp_sent = sorted(list(self.model.cp(self.raw).sents), key=len)[
                -1
            ]  # In case wrong analysis it picks the longest (sub)sentences recognized. This would need better treatment.
            test_label = str(cp_sent).replace(" ", "")
            init = [cp_sent]
            init_intervals = [(0, len(test_label))]
            edges = []
            it = 0
            while init != []:
                it += 1
                init_collect = []
                interval_collect = []
                for i, interval in zip(init, init_intervals):
                    ch_label = [str(ch).replace(" ", "") for ch in i._.children]
                    ch_int = []
                    for n in range(1, len(ch_label) + 1):
                        label_int = (
                            interval[0] + len("".join(list(ch_label[: n - 1])))
                        ), interval[0] + len("".join(list(ch_label[:n])))
                        ch_int.append(label_int)
                    node_list = [
                        (
                            (str(i).replace(" ", ""), interval),
                            (str(ch).replace(" ", ""), ch_i),
                        )
                        for ch, ch_i in zip(i._.children, ch_int)
                    ]
                    edges.append(node_list)
                    init_collect += i._.children
                    interval_collect += ch_int
                init = init_collect
                init_intervals = interval_collect

            edges = util_g.flatten(edges)
            return Tree(edges)

        else:
            return print(
                "No recognizable model name. Try any of these: sq, tr, ud, cp."
            )
