import nest_asyncio
import streamlit as st
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import gc
import matplotlib.colors as mcolors
import shutil
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sounddevice as sd
import soundfile as sf
import os
import requests
from bs4 import BeautifulSoup
from langdetect import detect
import re
import whisper
import yt_dlp
import subprocess
import datetime
import sys

#from text_mood_predictor import search_genius_lyrics,scrape_genius_lyrics,classify_long_text,load_emotion_model,load_whisper_model,delete_file,download_audio_youtube,transcribe_whisper,record_microphone,delete_microphone_file
from audio_functions import delete_file,download_audio_youtube,transcribe_whisper,record_microphone,delete_microphone_file
from cache_model_loading import load_emotion_model,load_whisper_model
from chunk_classification import classify_long_text
from lyrics_scraping_functions import search_genius_lyrics,scrape_genius_lyrics

# Load models and tokenizer
whisper_model = load_whisper_model()
model, tokenizer = load_emotion_model()

st.title("🎙️ Mood Detector — YouTube + Lyrics Scraper")

mode = st.radio("Select input:", ["YouTube URL", "🎤 Microphone"])

# After processing the song
if mode == "YouTube URL":
    url = st.text_input("Enter YouTube link")
    if url and st.button("Analyze Song"):
        with st.spinner("Downloading and processing audio..."):
            mp3_file, title = download_audio_youtube(url)
            st.write(f"Detected Title: **{title}**")

            lyrics = search_genius_lyrics(title)
            if lyrics:
                st.success("✅ Lyrics fetched from Genius")
                try:
                    lang = detect(lyrics[:300])
                    st.info(f"Detected language: {lang}")
                except:
                    st.warning("Could not detect language.")
            else:
                st.warning("Lyrics not found — transcribing audio instead...")
                lyrics = transcribe_whisper(mp3_file)  # Use MP3 file for transcription

            st.subheader("Lyrics/Text")
            st.text_area("Lyrics/Text", value=lyrics or "— No lyrics/text —", height=300)

            if not lyrics or len(lyrics.strip()) < 30:
                st.warning("Lyrics are too short to analyze.")
                st.stop()

            result = classify_long_text(lyrics, model, tokenizer)
            if result:
                label = result['label']
                score = result['score']
                st.subheader("Predicted Mood")
                st.success(label.capitalize())
                st.info(f"Confidence: {score:.2%}")
            else:
                st.warning("Could not determine mood.")

            # Provide download options for MP3
            st.download_button("⬇️ Download MP3", data=open(mp3_file, 'rb').read(), file_name=mp3_file, mime="audio/mp3")

            # Cleanup after the download is done
            delete_file(mp3_file)

        # Clean up memory by invoking garbage collection
        gc.collect()  # Attempt to free up memory

# After processing the microphone recording
elif mode == "🎤 Microphone":
    duration = st.slider("Recording duration (seconds)", 3, 15, 5)
    if st.button("🎙️ Start Recording"):
        with st.spinner("Recording..."):
            fn = record_microphone(duration)
        st.audio(fn)
        st.download_button("⬇️ Download Recording", data=open(fn, 'rb').read(), file_name=fn, mime="audio/wav")

        with st.spinner("Transcribing..."):
            text = transcribe_whisper(fn)

        st.subheader("Transcribed Text")
        st.text_area("Transcription", value=text or "— No transcription —", height=300)

        if text.strip():
            result = model(text)
            if result:
                label = result[0]['label']
                score = result[0]['score']
                st.subheader("Predicted Mood")
                st.success(label.capitalize())
                st.info(f"Confidence: {score:.2%}")
            else:
                st.warning("Could not determine mood.")

        # Provide download option for the microphone recording
        st.download_button("⬇️ Download Microphone Recording", data=open(fn, 'rb').read(), file_name=fn, mime="audio/wav")

        # Cleanup after processing is done
        delete_microphone_file(fn)
        
        # Clean up memory by invoking garbage collection
        gc.collect()  # Attempt to free up memory

# Enable asynchronous event loop for Streamlit
nest_asyncio.apply()

# --- Streamlit app title and description ---
st.title("🎵 Mood Predictor")
st.write("This app will predict the mood of a song based on audio features.")

# --- Load songs metadata CSV ---
# Load song metadata to create a song list for selection in the frontend
songs_df = pd.read_csv('../data/raw/features/metadata/all_songs.csv')

# List of song IDs to exclude from the app (based on testing or other criteria)
excluded_song_ids = ['48','3','116','386','47','324','646','691','634','329','656',
                     '622','152','637','8','174','769','149','278','54','488','996','113','1390','2006']

# Create a dataframe with songs that are excluded based on their song IDs
test_songs_df = songs_df[songs_df['song_id'].astype(str).isin(excluded_song_ids)]

# Generate a list of song options (title and artist) for the user to choose from
test_song_options = test_songs_df.apply(lambda row: f"{row['title']} — {row['artist']}", axis=1).tolist()
song_to_id = dict(zip(test_song_options, test_songs_df['song_id'].astype(str)))

# --- Load merged features and label encoder ---
# Load the preprocessed features and label encoder used for mood prediction
merged_df = pd.read_csv('../models/merged_features.csv')
le = LabelEncoder()
le.classes_ = np.load('../models/le_classes.npy', allow_pickle=True)

# Clean up the track ID column (stripping whitespace)
merged_df['track_id'] = merged_df['track_id'].astype(str).str.strip()

# Define columns to exclude from feature processing (like mood, track_id, etc.)
exclude_cols = ['track_id', 'mood', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std', 'valence_norm', 'arousal_norm']
feature_cols = [col for col in merged_df.columns if col not in exclude_cols]

# Create a dictionary of features for each track ID
features_dict = dict(zip(merged_df['track_id'], merged_df[feature_cols].values))

# --- Load saved scaler, graph, and model ---
# Load the scaler, edge index, and track order to ensure consistency during prediction
scaler = joblib.load('../models/scaler.save')
edge_index = torch.load('../models/edge_index.pt')
sorted_track_ids = np.load('../models/sorted_track_ids.npy').tolist()

# Set the computation device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Define the GraphSAGE model ---
# A simple GraphSAGE model for graph-based mood prediction
input_dim = len(feature_cols)
hidden_dim = 32
output_dim = len(le.classes_)

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.7):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Load the trained model
model = GraphSAGE(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load('../models/graphsage_model.pth', map_location=device))
model.eval()

# --- Prepare data for prediction ---
# Prepare the feature data, scale it, and convert it to tensor format for prediction
all_features = np.array([features_dict[tid] for tid in sorted_track_ids])
all_features_scaled = scaler.transform(all_features)
all_features_tensor = torch.tensor(all_features_scaled, dtype=torch.float).to(device)
graph_data = torch_geometric.data.Data(x=all_features_tensor, edge_index=edge_index.to(device))

# Ensure 'mood' column is present for model prediction
if 'mood' not in merged_df.columns:
    raise ValueError("The 'mood' column is missing from the DataFrame")

# Encode the mood labels for prediction
labels_encoded = le.transform(merged_df['mood'])
graph_data.y = torch.tensor(labels_encoded, dtype=torch.long).to(device)

# --- Mood prediction function ---
# Function to predict the mood of a given song based on track ID
def predict_mood(track_id: str):
    if track_id not in sorted_track_ids:
        return "Unknown (features not found)", None
    node_idx = sorted_track_ids.index(track_id)
    with torch.no_grad():
        out = model(graph_data)
        pred_class = out[node_idx].argmax(dim=0).item()
        pred_mood = le.inverse_transform([pred_class])[0]
    return pred_mood, node_idx

# --- Session state for mood image and statistics toggles ---
# Using session state to manage toggling the visibility of mood image and statistics
if 'show_mood_image' not in st.session_state:
    st.session_state.show_mood_image = False
if 'show_statistics' not in st.session_state:
    st.session_state.show_statistics = False

# --- Layout for buttons ---
# Create buttons for user interaction to show mood distribution or statistics
colA, colB = st.columns([1, 1])

with colA:
    mood_dist_button = st.button("Show Mood Distribution")
with colB:
    stats_button = st.button("Show Statistics")

# --- Toggle mood image visibility ---
if mood_dist_button:
    st.session_state.show_mood_image = not st.session_state.show_mood_image

# --- Toggle statistics visibility ---
if stats_button:
    st.session_state.show_statistics = not st.session_state.show_statistics

# --- Display mood distribution graph ---
if st.session_state.show_mood_image:
    st.image('../mood_distribution_graph.png', use_container_width=True)

# --- Display statistics about the graph ---
if st.session_state.show_statistics:
    G_temp = nx.Graph()
    G_temp.add_edges_from(edge_index.t().cpu().numpy())

    st.subheader("Graph Statistics")
    st.write(f"**Total Nodes:** {G_temp.number_of_nodes()}")
    st.write(f"**Total Edges:** {G_temp.number_of_edges()}")

    # Show the number of nodes per mood
    mood_counts = merged_df['mood'].value_counts().to_dict()
    st.write("**Nodes per Mood:**")
    for mood in ['happy', 'sad', 'calm', 'angry']:
        st.write(f"- {mood.capitalize()}: {mood_counts.get(mood, 0)}")

# --- Plot the graph with highlighted node ---
# Function to plot the graph with the selected song's node highlighted
def plot_graph_with_highlighted_node(node_idx, edge_index, predicted_mood, data):
    G = nx.Graph()
    G.add_edges_from(edge_index.t().cpu().numpy())
    neighbors = list(G.neighbors(node_idx))
    mood_colors = {'happy': 'yellow', 'sad': 'blue', 'angry': 'red', 'calm': 'green'}

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [200] * len(G.nodes)
    node_colors = ['lightblue'] * len(G.nodes)

    for i in range(len(G.nodes)):
        if i != node_idx and i not in neighbors:
            node_sizes[i] = 50

    node_sizes[node_idx] = 450
    node_colors[node_idx] = 'black'

    for neighbor in neighbors:
        mood = le.inverse_transform([data.y[neighbor].item()])[0]
        node_colors[neighbor] = mood_colors.get(mood, 'grey')

    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes,
            font_size=8, font_weight='bold', edge_color='grey', alpha=0.7, width=0.5)

    patches = [mpatches.Patch(color=color, label=label) for label, color in mood_colors.items()]
    plt.legend(handles=patches, loc='upper right')
    plt.title(f"Predicted Mood (black node): {predicted_mood}", fontsize=14)
    st.pyplot(plt)

# --- Song selection and mood prediction ---
# Dropdown to select a song from the test set
song_choice = st.selectbox("Choose a song from the test set:", test_song_options)
col1, col2 = st.columns([2, 1])

# Buttons to predict the mood and play the song
with col1:
    predict_button = st.button("Predict Mood")
with col2:
    play_button = st.button("Play Song")

# Retrieve the song file and play it
song_file = f'../data/raw/DEAM_audio/MEMD_audio/{song_to_id[song_choice]}.mp3'

if play_button:
    st.audio(song_file)

# Placeholder for graph plotting
plot_placeholder = st.empty()

# Predict mood when button is clicked
if predict_button:
    plot_placeholder.text("Plotting the graph live...")
    track_id = song_to_id[song_choice]
    predicted_mood, node_idx = predict_mood(track_id)
    st.success(f"The predicted mood for '{song_choice}' is: {predicted_mood}")
    plot_graph_with_highlighted_node(node_idx, edge_index, predicted_mood, graph_data)
    plot_placeholder.empty()
