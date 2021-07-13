import semiolog as slg

semiotic = slg.Cenematic("fr_wiki")

print(f"Numbers of cores: {semiotic.config.system.cpu_count}") 

print("\nSequential")
semiotic.vocab.build(vocab_size = 1627)

print("\nParallel Process")
semiotic.vocab.build(vocab_size = 1627, parallel=True, parallel_mode="process")

print("\nParallel Thread")
semiotic.vocab.build(vocab_size = 1627, parallel=True, parallel_mode="thread")