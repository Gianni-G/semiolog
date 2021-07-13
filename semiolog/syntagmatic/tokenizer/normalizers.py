import string
import unicodedata
import re

punctuation = "...—•…–’"

class Normalizer:
    """
    """
    def __init__(self) -> None:
        pass

    def normalize(self, input_string:str):
        pass

class disable:
    """
    Disable this step. It returns the input as output #TODO: maybe with the correct type for the pipeline
    """
    def __init__(self) -> None:
        pass

    def normalize(self, input_string:str):
        return input_string

class Sequence(Normalizer):
    def __init__(self,sequence:list) -> None:
        self.sequence = [eval(normalizer_sequence)() for normalizer_sequence in sequence]
    
    def __repr__(self) -> str:
        return f"Sequence({self.sequence})"

    def normalize(self, input_string: str):

        output = input_string
        for normalizer_component in self.sequence:
            output = normalizer_component.normalize(output)
        return output


class NFKC(Normalizer):
    """
    NFKC Normalizer using unicodedata module

    Args:
        Normalizer ([type]): [description]

    Returns:
        [type]: [description]
    """
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, input_string: str):
        return unicodedata.normalize("NFKC", input_string)

class NFKD(Normalizer):
    """
    NFKD Normalizer using unicodedata module

    Args:
        Normalizer ([type]): [description]

    Returns:
        [type]: [description]
    """
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, input_string: str):
        return unicodedata.normalize("NFKD", input_string)

class Lowercase(Normalizer):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, input_string: str):
        return input_string.lower()


class StripPunctuation(Normalizer):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, input_string: str):
        return input_string.translate(str.maketrans('', '', string.punctuation+punctuation))

class StripWhitespaces(Normalizer):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, input_string: str):
        return input_string.translate(str.maketrans('', '', string.whitespace))

class Foreign(Normalizer):
    """
    Replace Chinese, Japanese, Korean, Arabic, Cyrillic, Armenian, Hebrew, Syriac, Devanagari, Bengali characters
    """
    def __init__(self) -> None:
        super().__init__()
    
    def normalize(self, input_string: str):
        
        chars = re.compile(
            r"[\uac00-\ud7a3]|[\u4e00-\u9FFF]|[\u3040-\u30ff]|[\u0400-\u04FF]|[\u0500-\u052F]|[\u0530-\u058F]|[\u0590-\u05FF]|[\u0600-\u06FF]|[\u0700-\u074F]|[\u0750-\u077F]|[\u0980-\u09FF]|[\u0900-\u097F]"
            )
        
        
        return chars.sub("□", input_string)