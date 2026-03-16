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
