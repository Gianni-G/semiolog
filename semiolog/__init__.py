from typing import Union, Iterable, Dict, Any
from pathlib import Path

from .cenematic import Cenematic
from .corpus import Corpus

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