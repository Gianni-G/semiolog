from typing import Union, Iterable, Dict, Any
from pathlib import Path
from datasets import load_dataset #needed

from .cenematic import Cenematic #needed
from .corpus import Corpus #needed


def load(
    name: Union[str, Path],
    ) -> Cenematic:
    """
    Load a SemioLog model from an installed corpus or a local path

    RETURNS (Cenematic): the loaded semiolog object
    """

    return Cenematic(
        name
    )