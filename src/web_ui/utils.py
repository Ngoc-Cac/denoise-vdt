import librosa
import matplotlib.pyplot as plt
import numpy as np

from torchcodec.decoders import AudioDecoder


def draw_spec(
    file,
    n_fft = 2048,
    hop_len = 512,
    n_mels = 128,
    *,
    sr=16000,
    use_mel = True,
    name = None,
    cmap = 'viridis',
    ax=None,
):
    decoder = AudioDecoder(file, sample_rate=sr, num_channels=1)
    wf = decoder.get_all_samples().data.numpy()[0]

    if use_mel:
        spec = librosa.feature.melspectrogram(
            y=wf,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=n_mels
        )
        y_ax = 'mel'
    else:
        spec = np.abs(librosa.stft(wf, n_fft=n_fft, hop_length=hop_len)) ** 2
        y_ax = 'linear'
    spec_db = librosa.power_to_db(spec, ref=np.max)

    if ax is None:
        ax = plt.gca()

    spec_img = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_len,
        x_axis='time',
        y_axis=y_ax,
        cmap=cmap,
        ax=ax
    )
    if name:
        ax.set_title(name)

    return spec_img
