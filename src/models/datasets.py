import os
import pandas as pd
import scipy.io as scio
import datasets
import torch

from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from typing import Callable, Literal


_SUPPORTED_EXTS = (".wav", ".ogg", ".mp3")
_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../.."


def _filter_audio_files(files: list[str]):
    return list(filter(
        lambda file: os.path.splitext(file)[-1].lower() in _SUPPORTED_EXTS,
        files
    ))


class AudioDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        *,
        sample_rate: int | None = None,
        pad_max: bool = False
    ) -> None:
        super().__init__()

        self._preprocessor = preprocessor if preprocessor else lambda wf, _: wf

        self._decoders = [
            AudioDecoder(file, sample_rate=sample_rate)
            for file in _filter_audio_files(files) 
        ]
        self._cache = [None] * len(self._decoders)
        self._max_len = max(
            int(d.metadata.duration_seconds * d.metadata.sample_rate)
            for d in self._decoders
        ) if pad_max else None

    @staticmethod
    def collate_fn(batch: list[torch.Tensor]):
        # lengths = torch.tensor([sig.shape[-1] for sig in batch])
        padded_signals = torch.nested.nested_tensor(batch).to_padded_tensor(0)
        return padded_signals

    def __len__(self) -> int:
        return len(self._decoders)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._cache[index] is None:
            waveform = self._decoders[index].get_all_samples().data
            if self._max_len and waveform.shape[-1] < self._max_len:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, self._max_len - waveform.shape[-1])
                )

            self._cache[index] = self._preprocessor(
                waveform,
                self._decoders[index].metadata.sample_rate
            )

        return self._cache[index]


class SpeechTranscriptDataset:
    _SUPPORTED_DATASETS = (
        "doof-ferb/vlsp2020_vinai_100h", "doof-ferb/fpt_fosd"
    )

    def __init__(
        self,
        url: Literal["doof-ferb/vlsp2020_vinai_100h", "doof-ferb/fpt_fosd", None] = None,
        dataset: datasets.Dataset | None = None,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        *,
        subset_indices: list[int] | None = None,
        split: str = "train",
        **kwargs
    ):
        if url is not None:
            if url not in self._SUPPORTED_DATASETS:
                raise ValueError(f"Unsupported dataset {url}")

            self._dataset = datasets.load_dataset(url, split=split, **kwargs)
            if subset_indices:
                self._dataset = self._dataset.select(subset_indices)
        else:
            self._dataset = dataset

        if preprocessor is None:
            preprocessor = lambda wf, _: wf

        def process(batch: dict):
            return {
                "audio": [
                    preprocessor(
                        decoder.get_all_samples().data,
                        decoder.metadata.sample_rate
                    )
                    for decoder in batch['audio']
                ],
                "transcription": batch['transcription']
            }
        self._dataset.set_transform(process)

    @staticmethod
    def collate_fn(batch: list[dict]):
        audio = [item['audio'] for item in batch]
        transcripts = [item['transcription'] for item in batch]

        return AudioDataset.collate_fn(audio), transcripts

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._dataset[index]


class MSSNSDataset:
    def __init__(
        self,
        snr_levels: list[int | float],
        mssnsd_root: str | None = None,
        sample_strat: Literal["type", "cate", None] = "type",
        *,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        generator: torch.Generator | None = None
    ) -> None:
        if mssnsd_root is None:
            mssnsd_root = f"{_ROOT}/data/MS-SNSD"

        noise_dir = (f"{mssnsd_root}/noise_train", f"{mssnsd_root}/noise_test")
        files = ([f"{dir}/{file}" for file in os.listdir(dir)] for dir in noise_dir)
        audio_files = _filter_audio_files(sum(files, start = []))

        self._dataset = AudioDataset(audio_files, preprocessor)
        self._noise_weights = self._create_nosie_weights(audio_files, sample_strat)

        self._snr_levels = torch.tensor(snr_levels)

        self._generator = generator

    def _get_noise_type(self, filepath):
        return os.path.splitext(os.path.basename(filepath))[0].split('_')[0]

    def _create_nosie_weights(self, audio_files, sample_strat):
        categories = pd.read_csv(
            f"{_ROOT}/data/noise_categories.csv",
            index_col="noise_type"
        )

        if sample_strat == "type":
            weights = [
                categories.loc[self._get_noise_type(file), "count"]
                for file in audio_files
            ]
        elif sample_strat == "cate":
            cate_counts = categories.groupby("category")["count"].sum()
            weights = [
                cate_counts[categories.loc[
                    self._get_noise_type(file), "category"
                ]].item()
                for file in audio_files
            ]
        else:
            weights = [1] * len(audio_files)

        return 1 / torch.tensor(weights)

    def __len__(self) -> int:
        return len(self._dataset)

    def _match_length(self, waveform, target_len):
        len_diff = waveform.shape[-1] - target_len
        if len_diff > 0:
            start_idx = torch.randint(len_diff, size=(1,), generator=self._generator)
            waveform = waveform[..., start_idx : start_idx + target_len]
        elif len_diff < 0:
            waveform = torch.nn.functional.pad(waveform, (0, -len_diff))
        return waveform

    def get_samples(self, num_samples: int, target_len: int):
        sample_indices = torch.multinomial(
            self._noise_weights,
            num_samples,
            True,
            generator=self._generator
        )
        snr_indices = torch.randint(
            len(self._snr_levels),
            size=(num_samples,),
            generator=self._generator
        )

        return torch.stack(
            [
                self._match_length(self._dataset[idx], target_len)
                for idx in sample_indices
            ],
            dim=0
        ), self._snr_levels[snr_indices]


class FLAIRDataset:
    def __init__(
        self,
        flair_mat: str | None = None,
        preprocessor: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        *,
        generator: torch.Generator | None = None
    ) -> None:
        if flair_mat is None:
            flair_mat = f"{_ROOT}/data/data_FLAIR.mat"

        data = scio.loadmat(flair_mat)

        self._sample_rate = data["fs"][0, 0]
        self._rirs = torch.tensor(data["rirs"].transpose([1, 2, 0])).mean(dim=1, keepdim=True)

        self._preprocessor = preprocessor if preprocessor else lambda wf, _: wf
        self._generator = generator

    def __len__(self) -> int:
        return self._rirs.shape[0]

    def get_samples(self, num_samples: int):
        sample_indices = torch.randint(
            len(self),
            size=(num_samples,),
            generator=self._generator
        )

        return self._preprocessor(self._rirs[sample_indices], self._sample_rate)
