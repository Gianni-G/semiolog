class PostProcessor:
    """
    Base PostProcessor class
    """

    def __init__(self) -> None:
        pass

    def post_process(self, sequence):
        pass

class disable:
    """
    Disable this step. It returns the input as output #TODO: maybe with the correct type for the pipeline
    """

    def __init__(self) -> None:
        pass

    def post_process(self, sequence):
        return sequence

class WikiFR(PostProcessor):
    """
    Rules for cleaning the wikipedia corpus. Among others, only one occurrence of each sentence is kept, to avoid the statistical bias of titles and categories used by Wikipedia 

    Args:
        PostProcessor ([type]): [description]
    """
    def __init__(self) -> None:
        super().__init__()
    
    def post_process(self, sentences):

        return list({sent for sent in set(sentences)
                if
                all([
                not sent.startswith(("Catégorie:")),
                " " in sent,
                "│" not in sent,
                "|" not in sent,
                "─o" not in sent,
                ])
        })

class WikiEN(PostProcessor):
    """
    Rules for cleaning the wikipedia corpus. Among others, only one occurrence of each sentence is kept, to avoid the statistical bias of titles and categories used by Wikipedia 

    Args:
        PostProcessor ([type]): [description]
    """
    def __init__(self) -> None:
        super().__init__()
    
    def post_process(self, sentences):

        return list({sent for sent in set(sentences)
                if
                all([
                not sent.startswith(("Category:")),
                " " in sent,
                "│" not in sent,
                "|" not in sent,
                "─o" not in sent,
                ])
        })