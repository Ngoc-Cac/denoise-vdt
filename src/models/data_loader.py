import torch

from torch.utils.data import DataLoader
from torchaudio import functional as F

from .datasets import MSSNSDataset, FLAIRDataset


class NoiseAugmentLoader:
    def __init__(
        self,
        data_loader: DataLoader,
        snr_levels: list[int | float],
        snr_dataset: MSSNSDataset | None = None,
        rir_dataset: FLAIRDataset | None = None,
        snr_prob: float | None = None,
    ) -> None:
        if snr_prob is not None and not 0 < snr_prob < 1:
            raise ValueError("snr_prob must be between 0 and 1")

        self._data_loader = data_loader
        self._generator = data_loader.generator
        self.batch_size = self._data_loader.batch_size

        self._snr_dataset = (
            MSSNSDataset(generator=self._generator)
            if snr_dataset is None else snr_dataset
        )
        self._rir_dataset = (
            FLAIRDataset(generator=self._generator)
            if rir_dataset is None else rir_dataset
        )
        self._snr_prob = (
            len(self._snr_dataset) / (len(self._rir_dataset) + len(self._snr_dataset))
            if snr_prob is None else snr_prob
        )

        self._snr_levels = torch.tensor(snr_levels)

    def _apply_snr_noise(self, samples, snr_mask):
        snr_noises = self._snr_dataset.get_samples(
            snr_mask.sum(), samples.shape[-1]
        )
        snr_indices = torch.randint(
            len(self._snr_levels),
            size=(snr_noises.shape[0],),
            generator=self._generator
        )
        # reshape (B, 1) to match shape (B, ..., T) of the waveforms
        target_shape = [snr_noises.shape[0]] + list(samples.shape[1:-1])
        return snr_noises, snr_indices, F.add_noise(
            samples[snr_mask], snr_noises,
            self._snr_levels[snr_indices].view(target_shape)
        ).type_as(samples)

    def _apply_rir_noise(self, samples, rir_mask):
        rir_samples = self._rir_dataset.get_samples(rir_mask.sum())
        return rir_samples, F.fftconvolve(
            samples[rir_mask], rir_samples
        )[..., :samples.shape[-1]].type_as(samples)

    def __iter__(self):
        for batch in self._data_loader:
            samples, *other = batch if isinstance(batch, tuple) else (batch, ())
            batch_size = samples.shape[0]

            snr_mask = torch.bernoulli(
                torch.full((batch_size,), self._snr_prob),
                generator=self._generator
            ).bool()
            noisy_samples = torch.empty(samples.shape, dtype=samples.dtype)

            snr_noises, level_indices, noisy = self._apply_snr_noise(samples, snr_mask)
            noisy_samples[snr_mask] = noisy

            rir_samples, noisy = self._apply_rir_noise(samples, ~snr_mask)
            noisy_samples[~snr_mask] = noisy

            noise_dict = {
                "snr": snr_noises,
                "rir": rir_samples,
                "snr_mask": snr_mask,
                "snr_levels": self._snr_levels[level_indices]
            }

            yield samples, noisy_samples, noise_dict, *other
