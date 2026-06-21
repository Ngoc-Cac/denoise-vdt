import requests
import matplotlib.pyplot as plt

from .config import CACHE_DIR, ROOT_URL
from .utils import draw_spec


def clean_speech(audio_path, outfile: str = f"{CACHE_DIR}/clean.wav"):
    if audio_path is None:
        return None, ""

    with open(audio_path, "rb") as infile:
        response = requests.post(
            f"{ROOT_URL}/denoise",
            files={'audio': infile}
        )

    with open(outfile, "wb") as file:
        file.write(response.content)

    return outfile


def transcribe(audio_path: str):
    with open(audio_path, "rb") as infile:
        response = requests.post(
            f"{ROOT_URL}/transcribe",
            files={'audio': infile}
        )
    res = response.json()
    return res["transcript"] if 'transcript' in res else "<Error>"


def plot_spectrogram(audio_path, use_mel=True):
    if audio_path is None:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 4))
    img = draw_spec(
        audio_path,
        use_mel=use_mel,
        ax=ax
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()

    return fig
