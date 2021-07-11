from tqdm.notebook import tqdm

def chain_corpus(corpus_sents, normalizer):
    chain = ""
    for sent in corpus_sents:
        sent = normalizer.normalize(sent)
        sent = " ".join(sent)
        if sent !="":
            chain += " " + sent
    return chain.strip()