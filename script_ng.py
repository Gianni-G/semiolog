import semiolog as slg

semiotic = slg.Cenematic("en_bnc")

from datasets import Dataset
from tqdm import tqdm, trange

corpus = semiotic.corpus.train

corpus_norm = []
for sent in tqdm(corpus["text"]):
    corpus_norm.append(semiotic.syntagmatic.tokenizer.normalizer.normalize_str(sent))

corpus_norm = Dataset.from_dict({"text":corpus_norm})

for ng_n in trange(11,15):
    ng = slg.vocabulary.nGram.build(
        corpus=corpus_norm,
        n=ng_n,
        thres = 5,
        parallel=False,
        keep_in_memory=True,
        cpu_count = 24,
        )
    
    slg.util.save_file(ng,semiotic.paths.vocabulary / f"ngrams/{ng_n}_{thres}.json")