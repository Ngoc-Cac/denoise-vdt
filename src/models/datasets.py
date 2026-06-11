import os
import torch

from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from typing import Any, Callable


_SUPPORTED_EXTS = (".wav", ".ogg", ".mp3")
_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."



class AudioDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        preprocessor: Callable | None = None,
        *,
        sample_rate: int | None = None,
        pad: bool = False,
        seed: Any = None
    ) -> None:
        super().__init__()

        self._preprocessor = preprocessor if preprocessor else lambda x: x
        self._pad = pad

        files = filter(
            lambda file: os.path.splitext(file)[-1].lower() in _SUPPORTED_EXTS,
            files
        )
        self._decoders = [
            AudioDecoder(file, sample_rate=sample_rate)
            for file in files
        ]
        self._cache = [None] * len(self._decoders)
        self._max_len = max(
            int(d.metadata.duration_seconds * d.metadata.sample_rate)
            for d in self._decoders
        )

    def __len__(self):
        return len(self._decoders)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._cache[index] is None:
            waveform = self._decoders[index].get_all_samples().data
            if self._pad and waveform.shape[-1] < self._max_len:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, self._max_len - waveform.shape[-1])
                )

            self._cache[index] = self._preprocessor(waveform)

        return self._cache[index]


class MSSNSDataset:
    def __init__(
        self,
        mssnsd_root: str | None = None,
        *,
        generator: torch.Generator | None = None
    ) -> None:
        if mssnsd_root is None:
            mssnsd_root = f"{_ROOT}/data/MS-SNSD"

        noise_dir = (f"{mssnsd_root}/noise_train", f"{mssnsd_root}/noise_test")
        files = ([f"{dir}/{file}" for file in os.listdir(dir)] for dir in noise_dir)
        self._audio_files = sum(files, start = [])

        self._dataset = AudioDataset(self._audio_files)

        self._generator = generator

    def _match_length(self, waveform, target_len):
        len_diff = waveform.shape[-1] - target_len
        if len_diff > 0:
            start_idx = torch.randint(len_diff, size=(1,), generator=self._generator)
            waveform = waveform[..., start_idx : start_idx + target_len]
        elif len_diff < 0:
            waveform = torch.nn.functional.pad(waveform, (0, -len_diff))
        return waveform

    def get_samples(self, num_samples: int, target_len: int):
        sample_indices = torch.randint(
            len(self._dataset), size=(num_samples,), generator=self._generator
        )

        return torch.stack(
            [
                self._match_length(self._dataset[idx], target_len)
                for idx in sample_indices
            ],
            dim=0
        )
