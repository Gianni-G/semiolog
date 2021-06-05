# from transformers import pipeline
# from tqdm.notebook import tqdm, trange
from thinc.api import Config

# Loading the semiotic configs here is a roundabout not to load the unmasker at each execution of the chain_paradigm. This should be solved otherwise
# semiotic = Config().from_disk(paths.corpora / name / "config.cfg")


# unmasker = pipeline('fill-mask', model='distilbert-base-uncased',top_k=10)

def chain_paradigm(chain,semiotic, unmasker, thres=0):

    # unmasker = pipeline('fill-mask', model=semiotic.config["paradigm"]["model"],top_k=semiotic.config["paradigm"]["top_k"])

    sent_list = chain.split
    sent_mask = [" ".join([token if n!=i else "[MASK]" for n,token in enumerate(sent_list)]) for i in range(len(sent_list))]
    parads = [{i['token_str']:i['score'] for i in unmasker(sent) if i['score']>thres and i['token_str'] not in {
        ".",":",",","…","'","’","′",'"',"•",";","`","-","“","...","?","!","/","&","–"
        } } for sent in sent_mask]

    return parads