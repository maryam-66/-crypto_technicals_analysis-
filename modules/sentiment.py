import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF
import os

# ماژول تحلیل احساسات (فایل مجزا)
from modules import sentiment

# مسیر فونت‌ها
FONT_BOLD = "DejaVuSans-Bold.ttf"
FONT_REGULAR = "DejaVuSans.ttf"

# 📥 استخراج اخبار (شبیه‌سازی برای تست)
def extract_sample_headlines():
    return [
        "افزایش نرخ بهره توسط فدرال رزرو",
        "کاهش تورم در آمریکا در ماه گذشته",
        "تنش‌های ژئوپلیتیکی در خاورمیانه",
    ]

# 📄 تولید گزارش PDF
def generate_pdf(results, filename="sentiment_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", FONT_REGULAR, uni=True)
    pdf.add_font("DejaVu", "B", FONT_BOLD, uni=True)
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(200, 10, txt=get_display(arabic_reshaper.reshape("گزارش تحلیل احساسات")), ln=True, align='C')

    pdf.set_font("DejaVu", "", 12)
    for title, score in results:
        reshaped = arabic_reshaper.reshape(title)
        bidi_title = get_display(reshaped)
        pdf.multi_cell(0, 10, txt=f"{bidi_title}\nامتیاز احساسات: {score}\n")

    pdf.output(filename)

# 🎯 رابط کاربری Streamlit
def main():
    st.set_page_config(page_title="تحلیل احساسات اخبار اقتصادی", layout="centered")
    st.title("🧠 تحلیل احساسات اخبار اقتصادی")

    st.markdown("این ابزار با استفاده از NLP و مدل VADER احساسات اخبار اقتصادی را بررسی می‌کند.")

    if st.button("🔍 تحلیل احساسات اخبار"):
        with st.spinner("در حال بارگذاری اخبار..."):
            headlines = extract_sample_headlines()

        results = []
        for title in headlines:
            score = sentiment.analyze_sentiment(title)
            results.append((title, score))

        st.success("✅ تحلیل انجام شد!")
        st.write("### نتایج:")
        for title, score in results:
            st.write(f"📰 {title}")
            st.write(f"🔸 امتیاز احساسات: `{score}`")
            st.markdown("---")

        generate_pdf(results)
        with open("sentiment_report.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 دانلود گزارش PDF",
                data=pdf_file,
                file_name="sentiment_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

