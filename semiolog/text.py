from .syntagmatic import Chain, Tree
from .paradigmatic import ParadigmChain
from .typing import TypeChain

class Text:
    
    def __init__(self,input_chain,semiotic) -> None:

        self.chain = Chain(input_chain,semiotic)
        # semiotic.syntagmatic.tokenizer(self.chain)

        self.tree = Tree(self.chain.tree_tokens)
        
        # if semiotic.config.paradigmatic.load_pretrained != False:
        semiotic.paradigmatic.paradigmatizer(self.chain)

        self.parad_chain = ParadigmChain(self.chain)

        # semiotic.typing.typer(self.parad_chain)
        # self.type_chain = TypeChain(self.parad_chain)