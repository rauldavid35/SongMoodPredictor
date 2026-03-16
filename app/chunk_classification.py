import streamlit as st
import yt_dlp
import subprocess
import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import requests
from bs4 import BeautifulSoup
from langdetect import detect
import re
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import gc
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

# ---------- CHUNK-SAFE CLASSIFICATION WITH DEBUG ----------


def classify_long_text(text, model, tokenizer, max_chunk_tokens=64):
    tokens = tokenizer.tokenize(text)

    if not tokens:
        st.warning("No tokens found in text.")
        return None

    chunks = [" ".join(tokens[i:i + max_chunk_tokens]) for i in range(0, len(tokens), max_chunk_tokens)]

    if not chunks:
        st.warning("No valid chunks were generated from text.")
        return None

    # Track chunk sizes and emotions for plotting
    chunk_sizes = [len(chunk.split()) for chunk in chunks]
    chunk_emotions = []

    scores = {}
    for i, chunk in enumerate(chunks):
        decoded = tokenizer.convert_tokens_to_string(chunk.split())
        try:
            res = model(decoded, truncation=True)
            #st.code(f"[Chunk {i+1}] → {res}", language="text")
            if res and isinstance(res, list):
                label = res[0]['label']
                score = res[0]['score']
                chunk_emotions.append((f"Chunk {i+1}", label, score * 100))  # Store the chunk number, emotion, and score
                if label not in scores:
                    scores[label] = []
                scores[label].append(score)
        except Exception as e:
            st.warning(f"Error in chunk {i+1}: {e}")

    if not scores:
        st.warning("Model returned no predictions.")
        return None

    # Display emotions for each chunk in a table
    emotion_df = pd.DataFrame(chunk_emotions, columns=["Chunk", "Emotion", "Confidence (%)"])
    st.subheader("Emotions for Each Chunk")
    st.dataframe(emotion_df)

    # Calculate average scores for overall emotion
    averaged = {label: sum(vals) / len(vals) for label, vals in scores.items()}
    st.info(f"Averaged scores: {averaged}")

    best_label = max(averaged, key=averaged.get)

    return {"label": best_label, "score": averaged[best_label]}