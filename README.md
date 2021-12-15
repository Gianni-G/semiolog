# semiolog



## Initialization
    import semiolog as slg
    semiotic = slg.Cenematic("hf_tokenizers")
## Corpus

    semiotic.corpus.build(
        save = True,
        )

## Vocabulary

    semiotic.vocab.build(
        save = True,
        parallel = True,
        )

## Segmentation

