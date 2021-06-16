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