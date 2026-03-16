
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