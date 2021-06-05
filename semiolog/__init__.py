from typing import Union, Iterable, Dict, Any
from pathlib import Path

from . import cenematic

def load(
    name: Union[str, Path],
    ) -> cenematic.Cenematic:
    """
    Load a SemioLog model from an installed corpus or a local path

    RETURNS (Cenematic): the loaded semiolog object
    """

    return cenematic.Cenematic(
        name
    )


test_sent = "i have made my plans and i must stick to them"
