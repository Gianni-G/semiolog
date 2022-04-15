import semiolog as slg

semiotic = slg.Cenematic("abacus")

# print(f"Numbers of cores: {semiotic.config.system.cpu_count}")

# semiotic.corpus.build(
#     save = True,
#     )

# semiotic.vocab.build(
#     save = True,
#     parallel = True,
#     save_step = 1000,
#     )

# semiotic.syntagmatic.build()

semiotic.paradigmatic.build(
    checkpoints=True,
    min_token_length = 20,
    )