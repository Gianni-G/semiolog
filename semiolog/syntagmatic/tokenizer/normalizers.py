import string
import unicodedata

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





# # Constructors
# def build_tokenizer_component(config):
#     if config == "None":
#         component = lambda x:x
#     elif isinstance(config,list):
#         def component(input):
#             output = input
#             for func_str in config:
#                 func = eval(func_str)
#                 output = func(output)
#             return output
#     else:
#         component = eval(config)
#     return component