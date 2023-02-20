# Warning, the building of the vocabulary can be a computationally expensive task and take a considerable amount of time

import semiolog as slg

semiotic = slg.Cenematic("my_model")

semiotic.vocab.build(
    save = True,
    parallel = True,
    )