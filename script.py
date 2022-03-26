import semiolog as slg

semiotic = slg.Cenematic("en_bnc",requested_cpu = 32)

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

# semiotic.corpus.build(save = True)

semiotic.vocab.build(save = True, parallel = True, save_step=1000)