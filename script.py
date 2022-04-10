import semiolog as slg

semiotic = slg.Cenematic("en_bnc")

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

semiotic.syntagmatic.build()

# semiotic.paradigmatic.build(n_sents=200000)