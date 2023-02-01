"""
Enhancher wrappers of existing class.

At the moment the module contains only a wrapper of the sentiment analyzer 'StanfordCoreNLP'
"""
import json
import re
import socket
from typing import Any, Dict, List, Union

import numpy as np
from py4ai.data.model.text import Document
from nltk import TweetTokenizer
from typing_extensions import TypedDict

from py4ai.analytics.ml.core import Tokenizer
from py4ai.analytics.ml.core.enricher.enhancer.enhancer import DocumentEnhancher


class SentimentDict(TypedDict):
    """Sentiment dictionary typo the sentiment."""

    neg: float
    neu: float
    pos: float
    sentiment: str


class CustomTokenizer(Tokenizer):
    """Utility function to clean the text in a tweet by removing links and special characters using regex."""

    def __init__(
        self, tokenizer: bool = False, removePunctuation: bool = False
    ) -> None:
        """
        Initialize the class.

        :param tokenizer: whether or not tu use a tokenizer (TweetTokenizer by default)
        :param removePunctuation: whether to remove punctuation and links or only links
        """
        if removePunctuation:
            cleaner = self._removePunctuation
        else:
            cleaner = self._fallback

        if tokenizer:
            splitter = self._with_tokenizer
        else:
            splitter = None

        self.tokenizer = (
            lambda x: splitter(cleaner(x))
            if splitter is not None
            else cleaner(x).split()
        )

    _tknzr = None

    def _with_tokenizer(self, tweet: str) -> List[str]:
        """
        Tokenize tweet using tokenizer.

        :param tweet: tweet to tokenize
        :return: tokenized tweet
        """
        if self._tknzr is None:
            self._tknzr = TweetTokenizer()
        return self._tknzr.tokenize(tweet)

    @staticmethod
    def _removePunctuation(tweet: str) -> str:
        """
        Remove punctuation and links.

        :param tweet: tweet to clean
        :return: cleaned tweet
        """
        return re.sub(
            r"(\\$\\w+)|(@[A-Za-z0-9]+)|([^A-Za-z \t])|(\w+:\/\/\S+)|(http:\/\/)|(click here)|(RT)|(\n)",
            "",
            tweet,
        )

    @staticmethod
    def _fallback(tweet: str) -> str:
        """
        Remove links.

        :param tweet: tweet to clean
        :return: cleaned tweet
        """
        return re.sub(
            r"(\\$\\w+)|(@[A-Za-z0-9]+)|(\w+:\/\/\S+)|(RT)|(http:\/\/)|(click here)|(\n)",
            "",
            tweet,
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.

        :param text: text to tokenize
        :return: tokenized text
        """
        return self.tokenizer(text)


# TODO [ND] comment the whole class
class StanfordSentimentEnhancher(DocumentEnhancher):
    """Stanford Sentiment Enhancher."""

    @property
    def key(self) -> str:
        """
        Return key spec of the enhancer.

        :return: key spec of the enhancer
        """
        return "sentiment.stanford"

    def __init__(
        self,
        analyzer,  # TODO [ND] add typing
        tokenizer: Tokenizer = CustomTokenizer(tokenizer=False, removePunctuation=True),
    ):
        """
        Inizialize the class.

        :param analyzer: input analyzer. E.g. StanfordCoreNLP(host='http://localhost', port=9000)
        :param tokenizer: input tokenizer
        :raises OSError: Stanford service is not active
        """
        self.analyzer = analyzer
        if self.check_service_running(self.analyzer) != 0:
            raise OSError(
                "Stanford Service does not seems to be active on %s" % self.analyzer.url
            )

        self.tokenizer = tokenizer

    @staticmethod
    def check_service_running(nlp) -> int:  # TODO [ND] add input typing
        """
        Check if the analyzer is running.

        :param nlp: nlp service
        :return: 0 if service is up and running
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return sock.connect_ex((nlp.path_or_host.replace("http://", ""), nlp.port))

    def apply(self, document: Document) -> Document:
        """
        Enhanche the sentiment.

        :param document: original document
        :return: enhanced document
        """
        text = " ".join(self.tokenizer.tokenize(document.text))
        scores = self.get_sentiment(text)
        return document.addProperty(
            self.key, {"sentences": scores, "overall": self.calc_sentiment(scores)}
        )

    @staticmethod
    def _getdict(res: str) -> Dict[str, Any]:  # TODO [ND] specitfy return dict typing
        """
        Transform the string output of 'analyzer.annotate' in a dictionary.

        :param res: res
        :return: dictionary
        """
        r = res.split('"sentimentDistribution":[')

        for i in r[1:]:
            res = res.replace(i.split("]")[0], i.split("]")[0].replace("0,", "0."))

        return json.loads(res)

    def get_sentiment(self, text: str) -> List[SentimentDict]:
        """
        Get the sentiments list of a string text.

        :param text: text
        :return: list of sentiments
        """
        res = self._getdict(
            self.analyzer.annotate(
                text,
                properties={
                    "annotators": "sentiment",
                    "outputFormat": "json",
                    "timeout": 9999999,
                },
            )
        )

        lista = []

        for k_sentence in res["sentences"]:

            if (k_sentence["sentiment"] == "Verynegative") | (
                k_sentence["sentiment"] == "Negative"
            ):
                sent = "compound_negative"
            elif (k_sentence["sentiment"] == "Verypositive") | (
                k_sentence["sentiment"] == "Positive"
            ):
                sent = "compound_positive"
            else:
                sent = "compound_neutral"

            d = {
                "neg": k_sentence["sentimentDistribution"][0]
                + k_sentence["sentimentDistribution"][1],
                "neu": k_sentence["sentimentDistribution"][2],
                "pos": k_sentence["sentimentDistribution"][3]
                + k_sentence["sentimentDistribution"][4],
                "sentiment": sent,
            }

            lista.append(d)

        return lista

    @staticmethod
    def calc_sentiment(lista: List[SentimentDict]) -> Union[SentimentDict, Dict]:
        """
        Unify the sentiments.

        :param lista: list of sentiments
        :return: unified sentiments
        """
        if len(lista) == 0:
            return {}
        elif len(lista) == 1:
            return lista[0]
        else:
            keys = ["neg", "pos", "neu"]

            d = {key: np.mean([k[key] for k in lista]) for key in keys}

            max_key = max(d, key=lambda x: d[x])
            if max_key == "neg":
                sentiment = "compound_negative"
            elif max_key == "pos":
                sentiment = "compound_positive"
            else:
                sentiment = "compound_neutral"

            d.update({"sentiment": sentiment})

            return d
