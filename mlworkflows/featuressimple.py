import re

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

class SimpleSummaries(TransformerMixin, BaseEstimator):
    @staticmethod
    def columns():
        """ returns a list of the column names """
        return ['no_punct', 'number_words', 'mean_wl', 'max_wl', 'min_wl', 'pc_low_wl', 'pc_high_wl', 'upper', 'stop_words']
                        
    def __init__(self, lower_quantile=None, upper_quantile=None):
        self.lower_quantile = lower_quantile or 0.1
        self.upper_quantile = upper_quantile or 0.9
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = pd.DataFrame.from_records([self.standard_summary(doc) for doc in X])
        return features

    def standard_summary(self, row):
        """
        takes in text and returns 'simple' summaries.
        """
        no_punct = self.strip_punct(str(row))

        words = no_punct[0].split()

        number_words = len(words)

        word_length = [len(x) for x in words]

        mean_wl = sum(word_length)/number_words

        max_wl = max(word_length)
        min_wl = min(word_length)

        pc_hi_wl = np.percentile(word_length, self.upper_quantile * 100)
        pc_lo_wl = np.percentile(word_length, self.lower_quantile * 100)

        upper = sum([self.caps(x) for x in words])
        stop_words = sum([self.isstopword(x) for x in words])

        return dict(zip(SimpleSummaries.columns(), [no_punct[1], number_words, mean_wl, max_wl, min_wl, pc_lo_wl, pc_hi_wl, upper, stop_words]))

    def strip_punct(self, text):
        """
        takes in a document _doc_ and
        returns a tuple of the punctuation-free
        _doc_ and the count of punctuation in _doc_
        """
        return re.subn(r"""[!.><:;',@#~{}\[\]\-_+=£$%^&()?]""", "", str(text), count=0, flags=0)

    def caps(self, word):
        return not str(word).islower()

    def isstopword(self, word):
        return word in ENGLISH_STOP_WORDS
