import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# دانلود خودکار منابع nltk
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF
import streamlit as st
import os

# مسیر فونت‌ها
FONT_BOLD = "DejaVuSans-Bold.ttf"
FONT_REGULAR = "DejaVuSans.ttf"

# دانلود اخبار از investing.com (نمونه برای ایران)
def fetch_news():
    url = "https://www.investing.com/news/economy"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return ""

# استخراج اخبار (ساده‌سازی برای تست)
def extract_sample_headlines():
    return [
        "افزایش نرخ بهره توسط فدرال رزرو",
        "کاهش تورم در آمریکا در ماه گذشته",
        "تنش‌های ژئوپلیتیکی در خاورمیانه",
    ]

# تحلیل احساسات خبر
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score["compound"]

# تولید PDF از تحلیل
def generate_pdf(results, filename="sentiment_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", FONT_REGULAR, uni=True)
    pdf.add_font("DejaVu", "B", FONT_BOLD, uni=True)
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(200, 10, txt=get_display(arabic_reshaper.reshape("گزارش تحلیل احساسات")), ln=True, align='C')

    pdf.set_font("DejaVu", "", 12)
    for title, sentiment in results:
        reshaped_title = arabic_reshaper.reshape(title)
        bidi_title = get_display(reshaped_title)
        pdf.multi_cell(0, 10, txt=f"{bidi_title}\nامتیاز احساسات: {sentiment}\n")

    pdf.output(filename)

# رابط کاربری Streamlit
def main():
    st.set_page_config(page_title="تحلیل احساسات اخبار مالی", layout="centered")
    st.title("🧠 تحلیل احساسات اخبار اقتصادی")

    st.markdown("این ابزار با استفاده از NLP و مدل VADER احساسات اخبار اقتصادی را بررسی می‌کند.")

    if st.button("🔍 تحلیل احساسات اخبار"):
        with st.spinner("در حال بارگذاری اخبار..."):
            headlines = extract_sample_headlines()

        results = []
        for title in headlines:
            score = analyze_sentiment(title)
            results.append((title, score))

        st.success("✅ تحلیل انجام شد!")
        st.write("### نتایج:")
        for title, score in results:
            st.write(f"📰 {title}")
            st.write(f"🔸 امتیاز احساسات: `{score}`")
            st.markdown("---")

        generate_pdf(results)
        with open("sentiment_report.pdf", "rb") as pdf_file:
            btn = st.download_button(
                label="📄 دانلود گزارش PDF",
                data=pdf_file,
                file_name="sentiment_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
