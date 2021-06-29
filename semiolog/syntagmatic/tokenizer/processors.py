from ...util_g import subsequences
from ...functive import Functive
from math import log

import networkx as nx
from nltk.tokenize import sent_tokenize
        

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

    def build_graph_data(
        self,
        string: str,
        voc: dict
        ):
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
                            "freq": voc[subsequence_label],
                            "len": len(subsequence_label),
                        },
                    )
                )
        return edge_data

    def chain2seq(self,chain_sp, voc, v_rank = None):
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
        graph_data = self.build_graph_data(chain, voc)
        seg_graph_full = nx.DiGraph()
        seg_graph_full.add_edges_from(graph_data)
        # Construct weights
        for edge in seg_graph_full.edges:
            rank = voc_rank[seg_graph_full.edges[edge]["label"]]
            seg_graph_full.edges[edge]["weight"] = rank
        # Find best segmentation out of shortest path
        shortest_path = nx.shortest_path(seg_graph_full, 0, lSt, weight="weight")
        seg_sent = [
            seg_graph_full.edges[edge]["label"] for edge in subsequences(shortest_path, 2)
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


class TreeSLG(SequenceSLG):
    """
    """
    def __init__(self) -> None:
        super().__init__()

    def vocabulary_rank(self, voc):
        return super().vocabulary_rank(voc)

    def build_graph_data(self, string: str, voc: dict):
        return super().build_graph_data(string, voc)

    def chain2seq(self, chain_sp, voc, v_rank):
        return super().chain2seq(chain_sp, voc, v_rank=v_rank)


    # # Possible binary segmentation functions for binary_segment:
    def bin_seg_f_sum(self, l_units, r_units, voc):
        l_unit = l_units[-1]
        r_unit = r_units[-1]
        return (voc[l_unit] + voc[r_unit])

    def bin_seg_f_logLen(self, l_units, r_units, voc):
        l_unit = l_units[-1]
        r_unit = r_units[-1]
        return (log(voc[l_unit]+len(l_unit)) + log(voc[r_unit])+len(r_unit)) - (abs(len(l_units)-len(r_units)))


    def binary_segment(
        self,
        segmented_chain,
        voc,
        interval = None,
    ):
        chain = "".join(segmented_chain)
        if interval == None:
            interval = (0, len(chain))
        seg_sent_pairs = [
            [segmented_chain[:c], segmented_chain[c:]]
            for c in range(1, len(segmented_chain))
        ]
        cut_scores = []
        for pair in seg_sent_pairs:
            cut_scores.append([self.bin_seg_f_sum(pair[0], pair[1], voc), pair])
        best_cut = sorted(cut_scores, reverse=True)[0][1]
        seg_L = "".join(best_cut[0])
        seg_R = "".join(best_cut[1])
        return (
            (seg_L, (interval[0], interval[0] + len(seg_L))),
            (seg_R, (interval[1] - len(seg_R), interval[1])),
        )


    def chain2tree(self, chain_sp, voc, v_rank = None):
        chain = chain_sp.replace(" ", "")
        if v_rank == None:
            voc_rank = self.vocabulary_rank(voc)
        else:
            voc_rank = v_rank
        stack = [(chain, (0, len(chain)))]
        tree_data = []
        while len(stack) > 0:
            parent = stack[0]
            chain_to_seg = parent[0]
            stack = stack[1:]
            if len(chain_to_seg) == 2:
                l_child = (chain_to_seg[0], (parent[1][0], parent[1][0] + 1))
                r_child = (chain_to_seg[1], (parent[1][1] - 1, parent[1][1]))
            else:
                parent_seg = self.chain2seq(chain_to_seg, voc, voc_rank)
                children = self.binary_segment(parent_seg, interval=parent[1], voc=voc)
                l_child = children[0]
                r_child = children[1]
                for child in [l_child, r_child]:
                    if len(child[0]) > 1:
                        stack.append(child)
            tree_data.extend([(parent, l_child), (parent, r_child)])
        return tree_data
    
    def process(self, sequence: str, semiotic, is_pretokenized=False):

        tree_data = self.chain2tree(sequence,semiotic.vocab.freq)

        seq_tokens = sorted([child for head,child in tree_data if head[0] not in semiotic.vocab.freq and child[0] in semiotic.vocab.freq],key=lambda x:x[-1])

        tree_tokens = []
        for head,child in tree_data:
            child_token = Functive(child[0],child[1],None,semiotic)
            if child in seq_tokens:
                child_token.position = seq_tokens.index(child)
            child_token.head = Functive(head[0],head[1],None,semiotic)
            tree_tokens.append(child_token)




        return tree_tokens








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

class SentencesNLTK(Processor):
    """
    NLTK sentence tokenizer. Used for the normalization of initial datasets
    """
    def __init__(self) -> None:
        super().__init__()
    
    def process(self, sequence: str, semiotic=None, is_pretokenized=False):
        
        if is_pretokenized == False:
            return sent_tokenize(sequence)
        else:
            sentences = []
            for pre_token in sequence:
                nltk_sents = sent_tokenize(pre_token)
                for sent in nltk_sents:
                    sentences.append(sent)
            return sentences