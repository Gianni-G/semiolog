import semiolog as slg

semiotic = slg.Cenematic("fr_wiki", config_only=True)

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

semiotic.config.corpus.length = 40000

semiotic.corpus.build(save=True)

semiotic.vocab.build(save=True)