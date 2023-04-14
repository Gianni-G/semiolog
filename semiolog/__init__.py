from os import environ

# TODO: provide access to modification of these parameters
environ['TRANSFORMERS_OFFLINE'] = "1"
environ['HF_DATASETS_OFFLINE'] = "1"
environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
environ["TOKENIZERS_PARALLELISM"] = "true"
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hides TensorFlow message: This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations. To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

from typing import Union, Iterable, Dict, Any
from pathlib import Path
from datasets import load_dataset #needed

from .cenematic import Cenematic #needed
from .corpus import Corpus #needed