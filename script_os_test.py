import semiolog as slg

semiotic = slg.Cenematic("en_bnc_os_test")

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
    # n_sents=100000,
    epochs = 12,
    load_tokenized = True,
    checkpoints=True,
    # min_token_length = 3,
    )