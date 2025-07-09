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

# بررسی وجود vader_lexicon و دانلود در صورت نیاز
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# ساخت شیء تحلیلگر احساسات
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


# --- این قسمت را اگر در پروژه اصلی لازم داری استفاده کن ---
if __name__ == "__main__":
    sample_text = "I love this product! It's amazing."
    score = analyze_sentiment(sample_text)
    print(f"Sentiment score: {score}")
