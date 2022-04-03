import semiolog as slg

semiotic = slg.Cenematic("en_bnc")

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

semiotic.vocab.build(
    save = True,
    parallel = True,
    save_step = 1000,
    )

# semiotic.paradigmatic.build(load_tokenized=True, save_tokenized=True, n_sents=200000)