import requests
import matplotlib.pyplot as plt

from .config import CACHE_DIR, ROOT_URL
from .utils import draw_spec


def checkhealth():
    try:
        response = requests.get(f"{ROOT_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def fetch_available_models():
    return requests.get(f"{ROOT_URL}/models").json()


def load_denoising_ckpt(ckpt_name):
    res = requests.post(
        f"{ROOT_URL}/models",
        data={"ckpt_name": ckpt_name}
    )
    return res.status_code == 200


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


def transcribe(audio_path: str, models: list[str]):
    with open(audio_path, "rb") as infile:
        response = requests.post(
            f"{ROOT_URL}/transcribe",
            files={'audio': infile},
            data=[('models', i) for i in models]
        )
    if response.status_code != 200:
        return {name: "<Error>" for name in models}
    return response.json()


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
