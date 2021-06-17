from ... import util_g
from ... import util
from ...functive import Functive

import networkx as nx
        

class Processor:
    """
    Base Processor class
    """

    def __init__(self) -> None:
        pass

    def process(self, sequence: str, semiotic,is_pretokenized=False):
        pass

class disable:
    """
    Disable this step. It returns the input as output #TODO: maybe with the correct type for the pipeline
    """

    def __init__(self) -> None:
        pass

    def process(self, sequence: str, semiotic,is_pretokenized=False):
        return sequence #TODO: This should maybe return a list of Functives to preserve the flow of types in the tokenizer pipeline

class SequenceSLG(Processor):
    """
    """

    def __init__(self) -> None:
        self.zipf_factor = .135

    def vocabulary_rank(self,
        voc,
        ):
        # TODO: Zipf factor should (in principle) be computable following Mandelbrot
        # print('Building ranked vocabulary (voc_rank)')
        voc_rank = {k:(v+1)**self.zipf_factor for v,k in enumerate(voc.keys())}
        # print('Done!\n')
        return(voc_rank)

    def chain2seq(self,chain_sp, voc, v_rank = None): # , zipf_factor=.135):
        # Zipf's factor could (in principle) be computed following Mandelbrot
        chain = chain_sp.replace(" ", "")
        lSt = len(chain)
        if v_rank == None:
            voc_rank = self.vocabulary_rank(voc)
        else:
            voc_rank = v_rank
        for c in chain:
            if c not in voc:
                voc[c]=1
                voc_rank[c]=(len(voc)+1)**self.zipf_factor
        graph_data = util.build_graph_data(chain, voc)
        seg_graph_full = nx.DiGraph()
        seg_graph_full.add_edges_from(graph_data)
        # Construct weights
        for edge in seg_graph_full.edges:
            rank = voc_rank[seg_graph_full.edges[edge]["label"]]
            seg_graph_full.edges[edge]["weight"] = rank
        # Find best segmentation out of shortest path
        shortest_path = nx.shortest_path(seg_graph_full, 0, lSt, weight="weight")
        seg_sent = [
            seg_graph_full.edges[edge]["label"] for edge in util_g.subsequences(shortest_path, 2)
        ]
        return seg_sent
    
    def process(self, sequence: str, semiotic, is_pretokenized=False):
        
        segments = " ".join(self.chain2seq(sequence, semiotic.vocab.freq)).split()

        spans = []
        for i in range(len(segments)):
            start_i = len("".join(segments[:i]))
            end_i = start_i + len(segments[i])
            spans.append((start_i, end_i))

        tokens = [Functive(segment,span,position,semiotic) for position,(segment,span) in enumerate(zip(segments,spans))]

        tree_root = Functive(sequence, (0, len(sequence)), None, semiotic)
        tree_root.children = tokens

        for token in tokens:
            token.head = tree_root

        return tokens

class StripWhitespaces(Processor):
    def __init__(self) -> None:
        super().__init__()
    
    def process(self, sequence: str, semiotic, is_pretokenized):
        segments = sequence.split()

        spans = []
        for i in range(len(segments)):
            start_i = len("".join(segments[:i]))
            end_i = start_i + len(segments[i])
            spans.append((start_i, end_i))

        tokens = [Functive(segment,span,position,semiotic) for position,(segment,span) in enumerate(zip(segments,spans))]

        no_wspace_sequence = "".join(segments)
        tree_root = Functive(no_wspace_sequence, (0, len(no_wspace_sequence)), None, semiotic)
        tree_root.children = tokens

        for token in tokens:
            token.head = tree_root

        return tokens