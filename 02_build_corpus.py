    # Before running this script:

    # - Go through all the configurations in '/[my_model]/config.json' to modify default values according to your preferences for your model

    # - Place all the corresponding txt files of the corpus in '/[my_model]/corpus/original'
    
    import semiolog as slg

    semiotic = slg.Cenematic("my_model")
    
    semiotic.corpus.build(
        save = True,
        )