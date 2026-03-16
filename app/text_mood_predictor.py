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

# Cache the model loading
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model_core = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return pipeline("text-classification", model=model_core, tokenizer=tokenizer, device=-1), tokenizer

# Load models and tokenizer
whisper_model = load_whisper_model()
model, tokenizer = load_emotion_model()

# Path to FFMPEG
FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"

# ---------- LYRICS SCRAPING FUNCTIONS ----------

def search_genius_lyrics(song_title):
    search_url = f"https://genius.com/api/search/multi?per_page=5&q={song_title}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(search_url, headers=headers).json()
        for sec in resp.get("response", {}).get("sections", []):
            for hit in sec.get("hits", []):
                if hit.get("type") == "song":
                    title = hit["result"]["full_title"].lower()
                    url = hit["result"]["url"]

                    if any(word in title for word in ["türkçe", "translation", "traduction", "traducido", "übersetzung"]):
                        continue
                    if not song_title.lower().split()[0] in title:
                        continue

                    lyrics = scrape_genius_lyrics(url)
                    if lyrics:
                        try:
                            lang = detect(lyrics[:300])
                            if lang == "en":
                                return lyrics
                        except:
                            continue
    except Exception as e:
        st.error(f"Genius search error: {e}")

    return None

def scrape_genius_lyrics(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    conts = soup.select("div[class*='Lyrics__Container']") or soup.select("div.lyrics")
    lyrics = "\n".join(re.sub(r"\[.*?\]", "", div.get_text(separator="\n")) for div in conts)
    return lyrics.strip() if lyrics else None

# ---------- AUDIO FUNCTIONS ----------

def delete_file(file_path):
    """Function to delete a file"""
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        st.warning(f"File {file_path} does not exist")

def download_audio_youtube(url):
    ydl_opts = {'format': 'bestaudio/best', 'outtmpl': 'audio.%(ext)s',
                'noplaylist': True, 'ffmpeg_location': FFMPEG_PATH}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)
        title = info.get("title", "")
    
    # Saving audio as mp3
    mp3_out = f"audio_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp3"
    
    # Convert the audio file to mp3
    subprocess.run([FFMPEG_PATH, '-i', audio_file, mp3_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up the original downloaded file
    delete_file(audio_file)

    return mp3_out, title


def transcribe_whisper(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def record_microphone(sec):
    fs = 16000
    rec = sd.rec(int(sec * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    fn = f"mic_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(fn, rec, fs)
    return fn

def delete_microphone_file(fn):
    """Deletes the microphone recording file"""
    if os.path.exists(fn):
        os.remove(fn)
    else:
        st.warning(f"File {fn} does not exist")

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