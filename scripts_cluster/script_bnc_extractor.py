import nltk
from collections import Counter
from tqdm import tqdm
from semiolog.util import save_file

bnc = nltk.corpus.reader.bnc.BNCCorpusReader(root='models/BNC/Texts', fileids=r'[A-K]/\w*/\w*\.xml', lazy=False)
# bnc = nltk.corpus.reader.bnc.BNCCorpusReader(root='models/BNC/Texts', fileids=r'A/A0/\w*\.xml', lazy=False)

for n in range(3,5):

    # tag_counts = Counter()
    word_counts = Counter()

    for fileid in tqdm(bnc.fileids()):
        text = bnc.tagged_words(fileid)
        for tup in zip(*[text[i:] for i in range(n)]): 
            words, tags = list(zip(*tup))
            words = tuple([w.lower() for w in words])
            word_counts[words] += 1
            # tag_counts[tags] += 1

    for thres in [0,5,10]:

        # tag_counts = {" ".join(tup) : freq for tup, freq in tag_counts.most_common() if " " not in tup and freq>=thres}

        word_counts_save = {" ".join(tup) : freq for tup, freq in word_counts.most_common() if " " not in tup and freq>=thres}

        # save_file(tag_counts,f"wip/res_bnc/ng_t_{n}.json")
        save_file(word_counts_save,f"wip/res_bnc/ng_w_{n}_{thres}.json")