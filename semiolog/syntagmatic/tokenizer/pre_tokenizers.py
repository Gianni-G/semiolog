from typing import List
import networkx as nx

from tokenizers import (
    NormalizedString,
    PreTokenizedString,
)

from math import log

from semiolog.util import subsequences


class SequenceSLG:


    """
    """

    def __init__(self, semiotic) -> None:
        self.zipf_factor = .135
        self.semiotic = semiotic

        if self.semiotic.vocab.freq !=None:
            self.voc = self.semiotic.vocab.freq

            # TODO: Zipf factor should (in principle) be computable following Mandelbrot (or not?)
            self.voc_rank = {k:(v+1)**self.zipf_factor for v,k in enumerate(self.voc.keys())}

    def build_graph_data(
        self,
        string: str,
        voc: dict
        )-> List[tuple]:

        edge_data = []
        for beginning in range(0, len(string)):
            for end in range(beginning + 1, len(string) + 1):
                subsequence_label = string[beginning:end]
                if subsequence_label not in voc or subsequence_label == string:
                    continue
                edge_data.append(
                    (
                        beginning,
                        end,
                        {
                            "label": subsequence_label,
                        },
                    )
                )

        return edge_data

    def chain2seq(
        self, string:str
    ) -> List[tuple]:
        lSt = len(string)
        
        # If sent of len <2r, then return the interval
        if lSt<2:
            return [(0,lSt)]
        elif lSt == 2:
            return [(0,1),(1,2)]

        # If a character in the string not in vocab, add it
        for c in string:
            if c not in self.voc:
                self.voc[c]=1
                self.voc_rank[c]=(len(self.voc)+1)**self.zipf_factor

        graph_data = self.build_graph_data(string, self.voc)
        seg_graph_full = nx.DiGraph()
        seg_graph_full.add_edges_from(graph_data)

        # Construct weights
        for edge in seg_graph_full.edges:
            rank = self.voc_rank[seg_graph_full.edges[edge]["label"]]
            seg_graph_full.edges[edge]["weight"] = rank

        # Find best segmentation out of shortest path
        shortest_path = nx.shortest_path(seg_graph_full, 0, lSt, weight="weight")

        seg_offsets = subsequences(shortest_path, 2)

        return seg_offsets

    def SequenceSLG_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:

        seg_offsets = self.chain2seq(str(normalized_string))

        splits = []
        for start,end in seg_offsets:
            splits.append(normalized_string[start:end])

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):

        pretok.split(self.SequenceSLG_split)


class PMI_SLG:
    """
    Sequential segmentation maximizing the product of probabilities (which amounts to minizing mutual information between possible components of a given term)
    """

    def __init__(self, semiotic) -> None:
        self.semiotic = semiotic
        self.norm_factor = 1/semiotic.vocab.char_mass
        self.ng_prob = [{k:v*self.norm_factor for k,v in semiotic.vocab.__getattribute__(att).freq.items()} for att in [attribute for attribute in dir(semiotic.vocab) if attribute.startswith("ng")]]
        self.ng_prob = {len(next(iter(ng_i.keys()))):ng_i for ng_i in self.ng_prob}
        self.thres = semiotic.vocab.thres

    def build_graph_data(
        self,
        string: str,
        ng_prob: dict
        )-> List[tuple]:

        edge_data = []
        for beginning in range(0, len(string)):
            for end in range(beginning + 1, len(string) + 1):
                subsequence_label = string[beginning:end]
                if len(subsequence_label) not in ng_prob.keys() or subsequence_label not in ng_prob[len(subsequence_label)] or subsequence_label == string:
                    continue
                edge_data.append(
                    (
                        beginning,
                        end,
                        {
                            "label": subsequence_label,
                        },
                    )
                )

        return edge_data

    def chain2seq(
        self,
        string:str,
        ng_prob:dict,
        ) -> List[tuple]:
        
        lSt = len(string)

        # If sent of len <2r, then return the interval
        if lSt<2:
            return [(0,lSt)]
        elif lSt == 2:
            return [(0,1),(1,2)]

        # If a character in the string not in vocab, add it
        for c in string:
            if c not in ng_prob[1]:
                ng_prob[1][c]=0

        graph_data = self.build_graph_data(string, ng_prob)
        seg_graph_full = nx.DiGraph()
        seg_graph_full.add_edges_from(graph_data)

        # Construct weights

        if lSt not in ng_prob.keys():
            normalizer = 0
        else:
            normalizer = ng_prob[lSt].get(string,0)

        for edge in seg_graph_full.edges:
            label = seg_graph_full.edges[edge]["label"]
            score = ng_prob[len(label)].get(label,0)-normalizer
            
            if score <= self.thres:
                score = float("inf")
            else:
                score = -log(score)

            seg_graph_full.edges[edge]["weight"] = score

        ## Find best segmentation out of shortest path
        # def k_shortest_paths(G, source, target, k, weight=None):
        #     k_s = list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
        #     k_l = [[string[l:r] for l,r in subsequences(sp, 2)] for sp in k_s]
        #     k_sc = [[log(ng_prob[len(label)][label]) for label in k_] for k_ in k_l]
        #     return (k_s,k_l,k_sc)
        # k_short = k_shortest_paths(seg_graph_full, 0, lSt, 5)
        # shortest_path = nx.shortest_path(seg_graph_full, 0, lSt, weight="weight", method="dijkstra")

        shortest_path = nx.shortest_path(seg_graph_full, 0, lSt, weight="weight", method="bellman-ford")

        seg_offsets = subsequences(shortest_path, 2)

        return seg_offsets

    def PMI_SLG_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:

        seg_offsets = self.chain2seq(str(normalized_string),self.ng_prob)

        splits = []
        for start,end in seg_offsets:
            splits.append(normalized_string[start:end])

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):

        pretok.split(self.PMI_SLG_split)