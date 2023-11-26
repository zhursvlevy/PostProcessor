from typing import Any, Optional, List

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Processor:

    def __init__(self,
                 lemmatizer: Optional[Any] = None,
                 stop_words: Optional[List[str]] = None,
                 punctuation: Optional[re.Pattern] = None) -> None:
        
        if not lemmatizer:
            from pymystem3 import Mystem
            nltk.download('punkt')
            self.stem = Mystem()
        else:
            self.stem = lemmatizer

        if not stop_words:
            nltk.download('stopwords')
            self.stopwords = stopwords.words("russian")

        if not punctuation:
            self.punctuation = re.compile(r"[^а-яА-Яa-zA-Z-]")

    def __call__(self, text: str) -> str:

        tokens = word_tokenize(text, "russian")
        tokens = [token.lower() for token in tokens if token.lower() not in self.stopwords\
            and not re.findall(self.punctuation, token)]
        tokens = [self.stem.lemmatize(token)[0] for token in tokens]
        text = " ".join(tokens)

        return text
