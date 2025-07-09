import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import VaderConstants
from nltk.sentiment.util import register_punctuations

# لود کردن لغت‌نامه vader از فایل txt سفارشی
class CustomSentimentIntensityAnalyzer(SentimentIntensityAnalyzer):
    def __init__(self, lexicon_file_path=None):
        if lexicon_file_path is None:
            raise ValueError("مسیر فایل vader_lexicon.txt مشخص نشده است.")

        # لود کردن لغت‌نامه از فایل
        with open(lexicon_file_path, encoding='utf-8') as f:
            self.lexicon = self.make_lex_dict(f)
        
        self.constants = VaderConstants()
        self.constants.PUNCTUATION_LIST = register_punctuations()
        self.constants.BOOSTER_DICT = self.constants.BOOSTER_DICT
        self.constants.NEGATE = self.constants.NEGATE
        self.constants.SPECIAL_CASE_IDIOMS = self.constants.SPECIAL_CASE_IDIOMS
        self.constants.LEXICON = self.lexicon

# مسیر فایل vader_lexicon.txt
lexicon_path = os.path.join(os.path.dirname(__file__), "nltk_data", "sentiment", "vader_lexicon.txt")

# ساخت شی تحلیلگر با فایل شخصی‌سازی‌شده
sia = CustomSentimentIntensityAnalyzer(lexicon_file_path=lexicon_path)

# تابع تحلیل احساسات
def analyze_sentiment(text):
    """
    دریافت یک جمله و برگرداندن امتیاز احساسی آن (بین -1 تا 1)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    scores = sia.polarity_scores(text)
    return scores["compound"]

# تست سریع
if __name__ == "__main__":
    print(analyze_sentiment("This is absolutely fantastic!"))
