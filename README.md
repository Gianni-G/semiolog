# SemioLog



## Initialization
`01_create_empty_project.py`

    import semiolog as slg

    # Replace "my_model" with the name of your project 
    semiotic = slg.Cenematic("my_model")

    # Enter "Y" when prompted to create the model folder

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

