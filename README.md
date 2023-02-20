# semiolog

[![DOI](https://zenodo.org/badge/432172560.svg)](https://zenodo.org/badge/latestdoi/432172560)

`semiolog` is software package for the semiological analysis of corpora. It is intended for the use of the scientific community in the humanities. It is currently in alpha version.

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 839730.

Please consult the documentation [here](https://semiolog.readthedocs.io) (work in progress).

<!-- # Basic Procedures

The corresponding files can be found in the `scripts` folder in the project's repository.

## Initialization

`01_create_empty_project.py`

    import semiolog as slg

    # Replace "my_model" with the name of your project 
    semiotic = slg.Cenematic("my_model")

    # Enter "Y" when prompted to create the model folder

## Corpus Building

`02_build_corpus.py`

    # Before running this script:

    # - Go through all the configurations in '/[my_model]/config.json' to modify default values according to your preferences for your model

    # - Place the corresponding txt file of the corpus in '/[my_model]/corpus/original'
    
    import semiolog as slg

    semiotic = slg.Cenematic("my_model")
    
    semiotic.corpus.build(
        save = True,
        )

## Vocabulary Building

`03_build_vocabulary.py`

    # Warning, the building of the vocabulary can be a computationally expensive task and take a considerable amount of time

    import semiolog as slg

    semiotic = slg.Cenematic("my_model")

    semiotic.vocab.build(
        save = True,
        parallel = True,
        )

## Syntagmatic Analysis

`04_build_syntagmas.py`

    import semiolog as slg
    
    semiotic = slg.Cenematic("my_model")
    
    semiotic.syntagmatic.build()


## Paradigmatic Analysis

`05_build_paradigmatizer.py`

    import semiolog as slg

    semiotic = slg.Cenematic("hf_tokenizers")

    semiotic.paradigmatic.build() -->