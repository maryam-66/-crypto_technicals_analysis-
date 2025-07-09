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

# اضافه کردن مسیر دستی nltk_data به nltk.data.path
custom_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if custom_data_path not in nltk.data.path:
    nltk.data.path.append(custom_data_path)

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

# --- تست تابع (در صورت نیاز) ---
if __name__ == "__main__":
    sample_text = "I love this product! It's amazing."
    score = analyze_sentiment(sample_text)
    print(f"Sentiment score: {score}")
