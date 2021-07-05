from ..util import flatten
import networkx as nx
import graphviz as gv

class Tree:
    def __init__(self, tokens):
        self.edges = [((token.head.label,token.head.span),(token.label,token.span)) for token in tokens]
        self.nodes = set(flatten(self.edges))
        self.root = [node for node in self.nodes if node not in [r for l, r in self.edges]][
            0
        ]
        self.leaves = {node for node in self.nodes if node not in [l for l, r in self.edges]}
        self.non_terminals = set(self.nodes) - self.leaves

    def graph(self):
        seg_tree = nx.DiGraph()
        seg_tree.add_edges_from(self.edges)

        return seg_tree

    def plot(self, red=None, grey=None, fname=f"seg_graph"):

        seg_tree = self.graph()

        seg_tree_graph = gv.Digraph(name=fname)

        tree_nodes_list = [(str(i), l) for l, i in list(seg_tree.nodes)]
        for node in tree_nodes_list:
            seg_tree_graph.node(
                *node, color="white", fontsize="20", fontname="RobotoMono-Thin"
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
                        fontsize="20",
                        fontname="RobotoMono-Thin",
                    )

        if red == None:
            pass
        else:
            for node in [(str(i), l) for l, i in red]:
                if node in tree_nodes_list:
                    seg_tree_graph.node(
                        *node, color="red", fontsize="20", fontname="RobotoMono-Thin"
                    )

        return seg_tree_graph