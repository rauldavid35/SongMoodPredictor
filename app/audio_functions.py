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
from urllib.parse import urlparse, parse_qs
from cache_model_loading import load_emotion_model,load_whisper_model

# ---------- AUDIO FUNCTIONS ----------

# Load models and tokenizer
whisper_model = load_whisper_model()
model, tokenizer = load_emotion_model()

def get_video_urls(url):
    """Extrage o listă de link-uri dintr-un playlist sau returnează link-ul simplu."""
    
    # 1. Verificăm dacă link-ul conține un playlist
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    if 'list' in query_params:
        # Dacă da, extragem ID-ul playlist-ului și forțăm formatul de playlist
        playlist_id = query_params['list'][0]
        url = f"https://www.youtube.com/playlist?list={playlist_id}"

    # 2. Extragem datele cu yt-dlp
    ydl_opts = {'extract_flat': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        if 'entries' in info:
            # Este un playlist
            return [f"https://www.youtube.com/watch?v={entry['id']}" for entry in info['entries'] if entry.get('id')]
        else:
            # Este un singur video
            return [url]

# Path to FFMPEG
FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"

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