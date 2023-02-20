import semiolog as slg

semiotic = slg.Cenematic("abacus")

semiotic.paradigmatic.build(
    checkpoints=True,
    min_token_length = 20,
    )