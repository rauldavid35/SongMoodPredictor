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
import csv

#from text_mood_predictor import search_genius_lyrics,scrape_genius_lyrics,classify_long_text,load_emotion_model,load_whisper_model,delete_file,download_audio_youtube,transcribe_whisper,record_microphone,delete_microphone_file
from audio_functions import delete_file,download_audio_youtube,transcribe_whisper,record_microphone,delete_microphone_file,get_video_urls
from cache_model_loading import load_emotion_model,load_whisper_model
from chunk_classification import classify_long_text
from lyrics_scraping_functions import search_genius_lyrics,scrape_genius_lyrics


DB_FILE = "my_songs_db.csv"

def save_song_to_db(title, url, scores_dict):
    """Salvează titlul, scorurile emoțiilor și URL-ul la final într-un CSV, evitând duplicatele."""
    
    # 1. Verificăm dacă melodia există deja în fișier
    if os.path.isfile(DB_FILE):
        try:
            # Citim fișierul CSV existent
            df = pd.read_csv(DB_FILE)
            # Dacă titlul se află deja în coloana 'Title', oprim funcția
            if title in df['Title'].values:
                return False  # Returnăm False ca să știm că era deja acolo
        except Exception:
            pass # Dacă fișierul e corupt sau gol, ignorăm și trecem la salvare
            
    # 2. Dacă a trecut de verificarea de mai sus, înseamnă că e o melodie nouă. O salvăm!
    file_exists = os.path.isfile(DB_FILE)
    fieldnames = ['Title', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'URL']
    
    with open(DB_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        row = {
            'Title': title,
            'anger': scores_dict.get('anger', 0.0),
            'disgust': scores_dict.get('disgust', 0.0),
            'fear': scores_dict.get('fear', 0.0),
            'joy': scores_dict.get('joy', 0.0),
            'neutral': scores_dict.get('neutral', 0.0),
            'sadness': scores_dict.get('sadness', 0.0),
            'surprise': scores_dict.get('surprise', 0.0),
            'URL': url # Salvăm link-ul aici
        }
        writer.writerow(row)
        return True  # Returnăm True pentru a confirma că a fost salvată

# Load models and tokenizer
whisper_model = load_whisper_model()
model, tokenizer = load_emotion_model()

st.title("🎙️ Mood Detector — YouTube + Lyrics Scraper")

mode = st.radio("Select input:", ["YouTube URL", "🎤 Microphone"])

# After processing the song
if mode == "YouTube URL":
    url = st.text_input("Enter YouTube link (Video or Playlist)")
    if url and st.button("Analyze"):
        
        with st.spinner("Căutăm videoclipurile..."):
            video_urls = get_video_urls(url) # Apelăm noua funcție
            
        st.success(f"🔗 S-au găsit {len(video_urls)} melodii de procesat!")
        progress_bar = st.progress(0) # Inițializăm bara de progres
        
        # --- 1. INIȚIALIZĂM LISTELE PENTRU SESIUNEA CURENTĂ ---
        session_scores_list = []
        session_titles_list = []
        
        # Un mesaj temporar vizibil ca utilizatorul să știe că se lucrează
        status_text = st.empty()
        
        # --- 2. ASCUNDEM TOT PROCESUL TEHNIC ÎNTR-UN EXPANDER ---
        with st.expander("⚙️ Apasă aici pentru a vedea detaliile analizei (Versuri, Chunk-uri, Log-uri)", expanded=False):
            
            for i, vid_url in enumerate(video_urls):
                # Actualizăm textul vizibil din afara expander-ului
                status_text.info(f"⏳ Se analizează melodia {i+1} din {len(video_urls)}... Te rugăm să aștepți.")
                
                st.markdown("---")
                st.write(f"### 🔄 Detalii Melodia {i+1}")
                
                try:
                    mp3_file, title = download_audio_youtube(vid_url)
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
                        lyrics = transcribe_whisper(mp3_file)

                    with st.expander("Vezi Versurile / Textul transris"):
                        st.text_area(f"Lyrics/Text ({title})", value=lyrics or "— No lyrics/text —", height=150)

                    if not lyrics or len(lyrics.strip()) < 30:
                        st.warning(f"Versurile sunt prea scurte pentru {title}. Trecem peste.")
                        delete_file(mp3_file)
                        continue 

                    result = classify_long_text(lyrics, model, tokenizer)
                    if result:
                        label = result['label']
                        score = result['score']
                        all_scores = result.get('all_scores', {})
                        
                        st.success(f"**Predicted Mood:** {label.capitalize()} (Confidence: {score:.2%})")
                        
                        if all_scores:
                            # --- SALVĂM DATELE ÎN LISTELE TEMPORARE ---
                            session_scores_list.append(all_scores)
                            session_titles_list.append(title)

                            is_new = save_song_to_db(title, vid_url, all_scores)
                            if is_new:
                                st.info(f"💾 Salvat în baza de date!")
                            else:
                                st.warning(f"⚠️ Melodia se afla deja în baza de date. Am sărit peste salvare.")
                    else:
                        st.warning(f"Could not determine mood for {title}.")

                    # Cleanup după fiecare melodie
                    delete_file(mp3_file)
                    gc.collect() 
                    
                except Exception as e:
                    st.error(f"Eroare la procesarea melodiei {vid_url}: {e}")
            
                # Actualizăm bara de progres la finalul fiecărei iterații
                progress_bar.progress((i + 1) / len(video_urls))

        # Când bucla se termină, ștergem mesajul "Se analizează..." pentru un aspect curat
        status_text.empty()
        
        st.balloons()
        st.success("🎉 Analiza s-a terminat cu succes!")

        # --- 3. AICI ÎNCEPE SISTEMUL EXPERT DE RECOMANDĂRI ---
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(session_scores_list) > 0:
            st.markdown("---")
            st.header("🧠 Sistem Expert: Recomandările tale")
            st.write("Iată ce melodii din baza de date se potrivesc cu vibe-ul cerut (excluzând melodiile pe care tocmai le-ai introdus):")

            # Definim emoțiile pentru a extrage exact coloanele care ne interesează
            emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

            # Calculăm vibe-ul mediu al input-ului (ce a cerut utilizatorul acum)
            session_df = pd.DataFrame(session_scores_list)
            for col in emotion_cols:
                if col not in session_df.columns:
                    session_df[col] = 0.0
                    
            avg_vector = session_df[emotion_cols].mean().values.reshape(1, -1)

            # Afișăm graficul vibe-ului țintă
            st.subheader("📊 Profilul emoțional analizat (Ținta):")
            st.bar_chart(pd.DataFrame(avg_vector, columns=emotion_cols).T)

            # Căutăm recomandări în baza de date
            try:
                # Citim baza de date ACTUALIZATĂ
                db_df = pd.read_csv(DB_FILE)
                
                # Eliminăm din opțiunile de recomandare orice titlu a fost introdus ca input!
                candidates = db_df[~db_df['Title'].isin(session_titles_list)].copy()

                if len(candidates) >= 3:
                    # Calculăm Cosine Similarity între media input-ului și restul bazei de date
                    candidate_vectors = candidates[emotion_cols].values
                    similarities = cosine_similarity(avg_vector, candidate_vectors)[0]

                    # Adăugăm scorurile, sortăm și extragem Top 3
                    candidates['Similarity'] = similarities
                    top_recommendations = candidates.sort_values(by='Similarity', ascending=False).head(3)

                    st.subheader("✨ Top 3 Recomandări:")
                    for idx, row in top_recommendations.iterrows(): 
                        match_percentage = row['Similarity'] * 100
                        st.success(f"**{row['Title']}** (Compatibilitate: {match_percentage:.1f}%)")
                        
                        # VERIFICĂM DACĂ AVEM URL-UL ȘI AFIȘĂM CLIPUL YOUTUBE DIRECT
                        if 'URL' in candidates.columns and pd.notna(row['URL']) and str(row['URL']).strip() != "":
                            st.video(row['URL'])
                        else:
                            # Fallback pentru melodiile unde nu ai completat link-ul
                            search_query = str(row['Title']).replace(' ', '+')
                            st.markdown(f"[▶️ Click aici pentru a căuta '{row['Title']}' pe YouTube](https://www.youtube.com/results?search_query={search_query})")
                        
                        # Un mic grafic pentru a demonstra *de ce* a fost recomandată
                        with st.expander("Vezi detaliile emoționale ale recomandării"):
                            st.bar_chart(row[emotion_cols].to_frame().T)
                else:
                    st.warning("Nu există suficiente melodii diferite în baza de date pentru a face 3 recomandări. Mai analizează câteva piese noi!")
            except Exception as e:
                st.error(f"Eroare la generarea recomandărilor: {e}")

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
