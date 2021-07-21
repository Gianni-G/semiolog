import semiolog as slg

semiotic = slg.Cenematic("fr_wiki",requested_cpu = 4)

print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

print("\nSequential")
semiotic.vocab.build_new(vocab_size = -100, parallel=False)