class PreTokenizer:
    """
    Base PreTokenizer class
    """

    def __init__(self) -> None:
        pass

    def pre_tokenize(self, sequence:str):
        pass

class disable:
    """
    Disable this step. It returns the input as output #TODO: maybe with the correct type for the pipeline
    """

    def __init__(self) -> None:
        pass

    def pre_tokenize(self, sequence:str):
        return sequence


class SplitWhitespace(PreTokenizer):
    """
    """

    def __init__(self) -> None:
        super().__init__()

    def pre_tokenize(self, sequence: str):
        return sequence.split()