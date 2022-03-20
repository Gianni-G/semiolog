
# from . import normalizers
# from . import pre_tokenizers
# from . import processors
# from . import post_processors

from tokenizers.normalizers import NFKC, Lowercase, Replace

from tokenizers import (
    NormalizedString,
    PreTokenizedString,
    normalizers,
    Regex,
)

import networkx as nx
import string
from semiolog.util import subsequences

from typing import List

punctuation = "...—•…–’"

NormalizeSLG = normalizers.Sequence([NFKC(), Lowercase(), Replace(Regex(f"{[i for i in string.whitespace]}"),""),Replace(Regex(f"{[i for i in string.punctuation+punctuation]}"),"")])

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
        
        # In case of a character in the string not in vocab, add it
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