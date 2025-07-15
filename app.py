import streamlit as st
import librosa
import numpy as np
import requests
import io
import re
import pandas as pd
import matplotlib.pyplot as plt

# ============ CONFIG ============

def parse_drive_links(txt_file):
    """
    Reads a text file with lines:
    [Google Drive link], [year]
    and returns a list of dicts with filename, url, and year.
    """
    audio_files = []

    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        st.error(f"File {txt_file} not found. Please check the path.")
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split by comma
        parts = line.split(',')
        if len(parts) != 2:
            st.warning(f"Line format error: {line}")
            continue

        link, year_str = parts
        year = int(year_str.strip())

        # Extract Drive file ID
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
        if match:
            file_id = match.group(1)
            direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            filename = f"{file_id}.mp3"
            audio_files.append({
                'filename': filename,
                'url': direct_url,
                'year': year
            })
        else:
            st.warning(f"Could not extract file ID: {link}")

    return audio_files

# Load file list from local file in repo
audio_files = parse_drive_links('file_list.txt')

# ============ FUNCTIONS ============

def download_audio(url):
    """Downloads audio file from public Google Drive link."""
    response = requests.get(url)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        return None

def get_mean_pitch(y, sr):
    """Estimate mean pitch using librosa's piptrack."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitches = pitches[pitches > 0]
    if len(pitches) > 0:
        return np.mean(pitches)
    else:
        return 0

# ============ STREAMLIT APP ============

st.title("🎙️ Swedish Journalfilm Narrator Pitch Timeline")

data = []

for file in audio_files:
    filename = file['filename']
    year = file['year']
    # st.write(f"Processing: {filename} ({year})")

    audio_bytes = download_audio(file['url'])
    if audio_bytes:
        y, sr = librosa.load(audio_bytes, sr=None)
        mean_pitch = get_mean_pitch(y, sr)
        data.append({'filename': filename, 'year': year, 'mean_pitch': mean_pitch})
        # st.write(f"Mean pitch: {mean_pitch:.2f} Hz")
    else:
        st.warning(f"Could not download {filename}")

# Create DataFrame
df = pd.DataFrame(data)

# ============ PLOT ============

if not df.empty:
    st.subheader("Pitch Over Time")
    fig, ax = plt.subplots()
    ax.scatter(df['year'], df['mean_pitch'])
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Pitch (Hz)')
    ax.set_title('Narrator Mean Pitch by Year')
    st.pyplot(fig)
else:
    st.info("No data to plot yet.")
