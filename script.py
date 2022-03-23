import semiolog as slg

semiotic = slg.Cenematic("en_bnc_test",requested_cpu = 4)

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

semiotic.vocab.build(save = True, parallel = True, save_step=20)