
import streamlit as st
import tensorflow as tf
import numpy as np
import re
import nltk

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

model = tf.keras.models.load_model('my_distilbert_classifier.keras')




def predict_sentence_ai_probability(sentence):
    preds = model.predict([sentence])
    prob_ai = tf.sigmoid(preds[0][0]).numpy()
    return prob_ai

def predict_ai_generated_percentage(text, threshold=0.75):
    text=text+"."
    sentences = sent_tokenize(text)
    ai_sentence_count = 0
    results = []

    for sentence in sentences:
        prob = predict_sentence_ai_probability(sentence)
        is_ai = prob >= threshold
        results.append((sentence, prob, is_ai))
        if is_ai:
            ai_sentence_count += 1

    total_sentences = len(sentences)
    ai_percentage = (ai_sentence_count / total_sentences) * 100 if total_sentences > 0 else 0.0
    return ai_percentage, results

st.title("🧠 AI Content Detector")
st.markdown("This tool detects the percentage of **AI-generated content** in your input text based on sentence-level analysis.")

user_input = st.text_area("Paste your text here:", height=300)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        ai_percentage, analysis_results = predict_ai_generated_percentage(user_input)
        st.subheader("🔍 Sentence-level Analysis")
        for i, (sentence, prob, is_ai) in enumerate(analysis_results, start=1):
            label = "🟢 Human" if not is_ai else "🔴 AI"
            st.markdown(f"**{i}.** _{sentence}_\n\n→ **Probability AI:** `{prob:.2%}` → {label}")
        st.subheader("📊 Final Result")
        st.success(f"Estimated **AI-generated content**: **{ai_percentage:.2f}%**")
