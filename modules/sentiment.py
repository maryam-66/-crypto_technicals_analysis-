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
from nltk.sentiment import SentimentIntensityAnalyzer

# اضافه کردن مسیر دستی برای vader_lexicon
def setup_vader_lexicon():
    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../nltk_data"))
    if custom_path not in nltk.data.path:
        nltk.data.path.append(custom_path)

    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        raise RuntimeError("❌ خطا: فایل vader_lexicon یافت نشد! لطفاً مطمئن شوید فایل vader_lexicon.txt در مسیر زیر قرار دارد:\n" +
                           os.path.join(custom_path, "sentiment/vader_lexicon.txt"))

# راه‌اندازی Vader
setup_vader_lexicon()
sia = SentimentIntensityAnalyzer()

# تابع تحلیل احساسات
def analyze_sentiment(text):
    """
    دریافت یک جمله (string) و برگرداندن امتیاز احساسی آن.
    خروجی بین -1 (منفی) تا 1 (مثبت) است.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    scores = sia.polarity_scores(text)
    return scores["compound"]
