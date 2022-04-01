import semiolog as slg

semiotic = slg.Cenematic("en_bnc")

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

import datasets
tokenized_datasets = datasets.load_from_disk(semiotic.paths.paradigms / "tokenized")
tokenized_datasets = datasets.DatasetDict({
    "train":tokenized_datasets["train"].select(range(100000)),
    "dev": tokenized_datasets["dev"].select(range(5000))
})

semiotic.paradigmatic.build(dataset=tokenized_datasets, load_tokenized=True, save_tokenized=True)