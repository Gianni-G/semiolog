
# from . import normalizers
from .pre_tokenizers import (
    SequenceSLG,
    PMI_SLG,
)

from tokenizers.normalizers import NFKC, Lowercase, Replace

from tokenizers import (
    normalizers,
    Regex,
)

import string


punctuation = '...—•…–’‘·⁄′¿‐―'
punctuations = (string.punctuation+punctuation).replace("'","").replace('"','').replace('[','').replace(']','')

NormalizeSLG = normalizers.Sequence([
    NFKC(),
    Lowercase(),
    Replace(Regex(f"{[i for i in string.whitespace]}"),""),
    Replace(Regex("["+punctuations+"]"),"-"),
    Replace(Regex("[\'\"\[\]\-]"),""),
    ])

Normalize_w_spacesSLG = normalizers.Sequence([
    NFKC(),
    Lowercase(),
    # Replace(Regex(f"{[i for i in string.whitespace]}"),""),
    Replace(Regex("["+punctuations+"]"),"-"),
    Replace(Regex("[\'\"\[\]\-]"),""),
    ])

Normalize_menoSLG = normalizers.Sequence([
    NFKC(),
    Lowercase(),
    # Replace(Regex(f"{[i for i in string.whitespace]}"),""),
    Replace(Regex("["+punctuations+"]"),"-"),
    Replace(Regex("[\'\"\[\]\-]"),""),
    Replace(Regex("j$"),"i"),
    ])

# TODO: "NormalizeSLG" should be replaced by generic normalizers to be added to a standard HF normalizer sequence
# Below is the model to use for doing it
#
# class CustomNormalizer:
#     def normalize(self, normalized: NormalizedString):
#         # Most of these can be replaced by a `Sequence` combining some provided Normalizer,
#         # (ie Sequence([ NFKC(), Replace(Regex("\s+"), " "), Lowercase() ])
#         # and it should be the prefered way. That being said, here is an example of the kind
#         # of things that can be done here:
#         normalized.nfkc()
#         normalized.filter(lambda char: not char.isnumeric())
#         normalized.replace(Regex("\s+"), " ")
#         normalized.lowercase()
#
# # The custom normalizer should be loaded as follows:
#
# tok.normalizer = Normalizer.custom(CustomNormalizer())