import streamlit as st
import librosa
import numpy as np
import requests
import io
import re
import pandas as pd
import matplotlib.pyplot as plt

# ============ CONFIG ============

# List your audio files here:
# Replace with your actual public Google Drive direct download URLs
audio_files = [
    {
        'filename': 'Kino37.1_1932.0.mp3',
        'url': 'https://drive.google.com/uc?export=download&id=1otg3BlM_I-ovQ3_csUYK2aFEWe_uSQ1X'
    },
    {
        'filename': 'Kino44.1_1942.0.mp3',
        'url': 'https://drive.google.com/uc?export=download&id=1qA4V3IglpOKSKM1CIWRx5JdxAislXTcx'
    },
    # Add more here...
]

# ============ FUNCTIONS ============

def extract_year(filename):
    """Extracts a 4-digit year from the filename."""
    match = re.search(r'_(\d{4})\.', filename)
    if match:
        return int(match.group(1))
    else:
        return None

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
    year = extract_year(filename)
    st.write(f"Processing: {filename} ({year})")

    audio_bytes = download_audio(file['url'])
    if audio_bytes:
        y, sr = librosa.load(audio_bytes, sr=None)
        mean_pitch = get_mean_pitch(y, sr)
        data.append({'filename': filename, 'year': year, 'mean_pitch': mean_pitch})
        st.write(f"Mean pitch: {mean_pitch:.2f} Hz")
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
