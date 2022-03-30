import semiolog as slg

semiotic = slg.Cenematic("en_bnc")

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

# semiotic.paradigmatic.build()

from datasets import DatasetDict, Dataset

dataset = DatasetDict({
    "train": Dataset.from_dict(semiotic.corpus.dataset["train"][:1000]),
    "dev": Dataset.from_dict(semiotic.corpus.dataset["dev"][:100]),
    "test": Dataset.from_dict(semiotic.corpus.dataset["test"][:100])
    })
semiotic.paradigmatic.build(dataset = dataset, load_tokenized=True, save_tokenized=True)