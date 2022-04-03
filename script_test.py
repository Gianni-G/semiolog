import semiolog as slg

semiotic = slg.Cenematic("en_bnc_test")

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

semiotic.paradigmatic.build(load_tokenized=True, save_tokenized=True, n_sents=200000)