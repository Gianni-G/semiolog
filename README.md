# SemioLog



## Initialization

`01_create_empty_project.py`

    import semiolog as slg

    # Replace "my_model" with the name of your project 
    semiotic = slg.Cenematic("my_model")

    # Enter "Y" when prompted to create the model folder

## Corpus

`02_build_corpus.py`

    # Before running this script:

    # - Go through all the configurations in '/[my_model]/config.json' to modify default values according to your preferences for your model

    # - Place the corresponding txt file of the corpus in '/[my_model]/corpus/original'
    
    import semiolog as slg

    semiotic = slg.Cenematic("my_model")
    
    semiotic.corpus.build(
        save = True,
        )

## Vocabulary

    semiotic.vocab.build(
        save = True,
        parallel = True,
        )

## Segmentation

