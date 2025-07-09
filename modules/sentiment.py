import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# تابع ساخت دیکشنری لغت‌نامه از فایل vader_lexicon.txt
def load_custom_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip() or line.startswith('#'):
                continue
            word, measure = line.strip().split('\t')[0:2]
            lexicon[word] = float(measure)
    return lexicon


# آدرس فایل vader_lexicon.txt (باید در پروژه موجود باشد)
custom_lexicon_path = os.path.join(os.path.dirname(__file__), 'nltk_data', 'sentiment', 'vader_lexicon.txt')

# بارگذاری لغت‌نامه سفارشی
custom_lexicon = load_custom_lexicon(custom_lexicon_path)

# ساخت تحلیلگر احساسات با لغت‌نامه دلخواه
sia = SentimentIntensityAnalyzer()
sia.lexicon.update(custom_lexicon)


# تابع تحلیل احساسات
def analyze_sentiment(text):
    """
    دریافت یک جمله و بازگرداندن امتیاز احساسی آن بین -1 تا 1.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    score = sia.polarity_scores(text)["compound"]
    return score


# --- تست در صورت اجرای مستقیم ---
if __name__ == "__main__":
    print(analyze_sentiment("I really love this amazing tool!"))
