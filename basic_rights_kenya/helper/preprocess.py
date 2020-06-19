import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter

import re, string, unicodedata
import random

from bs4 import BeautifulSoup

# contractions
from contractions import CONTRACTION_MAP
from slangs import SLANGS_MAP

import inflect
import unidecode
import emoji

import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.lemmatizer import Lemmatizer

from spellchecker import SpellChecker
from wordcloud import WordCloud, STOPWORDS


from textblob import TextBlob

import nltk


## initilize glocal variables
nlp = spacy.load('en_core_web_md')
spell = SpellChecker()
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


# preprocessing class
class PreProcess:
    """Class to pre process text"""

    def correct_spellings(self, words):
        corrected_text = []
        misspelled_words = spell.unknown(words)
        for word in words:
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)

        return corrected_text

    def spell_checker(self, text):
        text = TextBlob(text)
        return text.correct()

    def spacy_tokenization_lemma(self, text):
        text_nlp = nlp(text)
        token_list = []
        for token in text_nlp:
            if token.lemma_ == '-PRON-':
                token_list.append(token.text)
            else:
                token_list.append(token.lemma_)

        return token_list

    def _remove_punct(self, text):
        punctuations = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        for x in text.lower():
            if x in punctuations:
                text = text.replace(x, "")

        return text

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_words.append(self._remove_punct(word))

        return new_words

    # convert emojis
    def convert_emoji(self, text):
        text = emoji.demojize(text)
        return text


    # remove html tags
    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text


    # remove URL from text
    def remove_url(self, words):
        """Remove URLs from a sample string"""
        # url_rule = r'(?P<url>https?://[^\s]+)'
        word_clean = []
        for word in words:
            text = re.sub(r"http\S+", "", word)
            if text:
                word_clean.append(text)
        return word_clean

    def unslang(self, text):
        """Replace slangs in text"""
        for slang in SLANGS_MAP.keys():
            text = text.replace(slang, text)

        return text

    def expand_contractions_old(self, words):
        words_expand = []
        for word in words:
            if word in CONTRACTION_MAP.keys():
                words_expand.append(CONTRACTION_MAP[word])
            else:
                words_expand.append(word)

        text_expand = ' '.join(words_expand)
        return text_expand

    def expand_contractions(self, text):
        "replace contractions"
        for contraction in CONTRACTION_MAP.keys():
            print(contraction)
            text = text.replace(contraction, text)
        return text


    ##--- remove special characters
    def remove_special_characters(self, text):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text


    # remove remianing non ascii characters
    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words


    ###---- Remove stop words

    def remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in spacy_stopwords:
                new_words.append(word)
        return new_words


    def replace_compound_words(self, text):
        """Break compound words"""
        text = re.sub('/', ' / ', text)
        text = re.sub('-', ' - ', text)
        text = re.sub('_', ' _ ', text)

        return text


    # L33T vocabulary (SLOW)
    # https://simple.wikipedia.org/wiki/Leet
    # Local (only unknown words)
    def convert_leet(self, word):
        # basic conversion 
        word = re.sub('0', 'o', word)
        word = re.sub('1', 'i', word)
        word = re.sub('3', 'e', word)
        word = re.sub('\$', 's', word)
        word = re.sub('\@', 'a', word)
        return word

    def normalization(self, text):
        text = text.lower()
        text = self.replace_compound_words(text)
        text = self.spell_checker(text)
        text = self.convert_emoji(text)
        text = self.strip_html_tags(text)
        text = self.expand_contractions(text)
        text = self.unslang(text)

        # Tokenize
        words = self.spacy_tokenization_lemma(text)
        words = self.correct_spellings(words)
        words = self.remove_stopwords(words)
        words = self.remove_url(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)

        words = ' '.join(words)
        words = words.strip().lower()

        return words


if __name__ == "__main__":
    text = ["I've been iin this shit for more than 10 years", "go f**k your ars-hole"]
    norm = PreProcess()
    print(norm.expand_contractions_old(text[0]))
