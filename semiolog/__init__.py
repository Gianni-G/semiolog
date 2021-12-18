from os import environ

# TODO: provide access to modification of these parameters
environ['TRANSFORMERS_OFFLINE'] = "0"
environ['HF_DATASETS_OFFLINE'] = "0"

from typing import Union, Iterable, Dict, Any
from pathlib import Path
from datasets import load_dataset #needed

from .cenematic import Cenematic #needed
from .corpus import Corpus #needed