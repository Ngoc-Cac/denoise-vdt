import torch

from torch.utils.data import DataLoader
from torchaudio import functional as F

from .datasets import MSSNSDataset


class NoiseAugmentLoader:
    def __init__(
        self,
        data_loader: DataLoader,
        snr_levels: list[int | float],
        noise_dataset: MSSNSDataset | None = None
    ) -> None:
        self._data_loader = data_loader
        self._generator = data_loader.generator
        self.batch_size = self._data_loader.batch_size

        self._noise_dataset = (
            MSSNSDataset(generator=self._generator)
            if noise_dataset is None else noise_dataset
        )
        self._snr_levels = torch.tensor(snr_levels)

    def __iter__(self):
        for samples in self._data_loader:
            noises = self._noise_dataset.get_samples(
                self.batch_size, samples.shape[-1]
            )

            snr_indices = torch.randint(
                len(self._snr_levels),
                size=(self.batch_size,),
                generator=self._generator
            )
            # reshape (B, 1) to match shape (B, ..., T) of the waveforms
            target_shape = samples.shape[:-1]

            yield samples, noises, F.add_noise(
                samples, noises,
                self._snr_levels[snr_indices].view(target_shape)
            )
