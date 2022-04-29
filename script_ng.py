import semiolog as slg

semiotic = slg.Cenematic("en_bnc")

from datasets import Dataset
from tqdm import tqdm, trange

corpus = semiotic.corpus.train

corpus_norm = []
for sent in tqdm(corpus["text"]):
    corpus_norm.append(semiotic.syntagmatic.tokenizer.normalizer.normalize_str(sent))

corpus_norm = Dataset.from_dict({"text":corpus_norm})

thres = 5
for ng_n in trange(15,27):
    ng = slg.vocabulary.nGram.build(
        corpus=corpus_norm,
        n=ng_n,
        thres = thres,
        parallel=False,
        keep_in_memory=True,
        cpu_count = 24,
        )
    
    slg.util.save_file(ng,semiotic.paths.vocabulary / f"ngrams/{ng_n}_{thres}.json")