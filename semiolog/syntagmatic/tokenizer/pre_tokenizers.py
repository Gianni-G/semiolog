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

class Layout(PreTokenizer):
    """
    This pre-tokenizer splits a string at its linebreaks and strips the resulting elements from intial and ending whitespaces. The goal is to clean raw input with layout (typically from Wikipedia), for a further tokenization in sentences. All this is needed for normalizing initial datasets
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    def pre_tokenize(self, sequence: str):
        return [paragraph.strip() for paragraph in sequence.split("\n")]