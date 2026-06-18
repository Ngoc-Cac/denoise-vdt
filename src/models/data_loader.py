import torch

from torch.utils.data import DataLoader
from torchaudio import functional as F

from .datasets import MSSNSDataset, FLAIRDataset


class NoiseAugmentLoader:
    def __init__(
        self,
        data_loader: DataLoader,
        snr_dataset: MSSNSDataset,
        rir_dataset: FLAIRDataset | None = None,
        noise_weights: list | None = None,
        *,
        resample: bool = True
    ) -> None:
        self._data_loader = data_loader
        self._generator = data_loader.generator
        self.batch_size = self._data_loader.batch_size

        self._snr_dataset = snr_dataset
        self._rir_dataset = (
            FLAIRDataset(generator=self._generator)
            if rir_dataset is None else rir_dataset
        )
        self._noise_weights = (
            torch.ones(3) if noise_weights is None else
            torch.tensor(noise_weights)
        )

        self._cache = None if resample else [None] * len(self._data_loader)

    def _apply_snr_noise(self, samples, snr_mask):
        snr_noises, snr_levels = self._snr_dataset.get_samples(
            snr_mask.sum(), samples.shape[-1]
        )

        # reshape (B, 1) to match shape (B, ..., T) of the waveforms
        target_shape = [snr_noises.shape[0]] + list(samples.shape[1:-1])
        return snr_noises, snr_levels, F.add_noise(
            samples[snr_mask], snr_noises,
            snr_levels.view(target_shape)
        ).type_as(samples)

    def _apply_rir_noise(
        self,
        samples: torch.Tensor,
        rir_mask,
        eps: float = 1e-7
    ):
        rir_samples = self._rir_dataset.get_samples(rir_mask.sum())
        samples = samples[rir_mask]

        noisy = F.fftconvolve(
            samples, rir_samples
        )[..., :samples.shape[-1]].type_as(samples)

        dims = tuple(range(1, samples.ndim))
        clean_rms = (samples ** 2).mean(dims, True).sqrt() + eps
        noisy_rms = (noisy ** 2).mean(dims, True).sqrt() + eps

        return rir_samples, noisy * (clean_rms / noisy_rms)

    def _augment(self, batch):
        samples, *other = batch if isinstance(batch, tuple) else (batch, ())
        batch_size = samples.shape[0]

        # 0: rir | 1: snr | 2: both
        noise_mask = torch.multinomial(
            self._noise_weights,
            num_samples=batch_size,
            replacement=True,
            generator=self._generator
        )
        noisy_samples = samples.clone()

        rir_mask = (noise_mask == 0) | (noise_mask == 2)
        rir_samples, noisy = self._apply_rir_noise(noisy_samples, rir_mask)
        noisy_samples[rir_mask] = noisy

        snr_mask = (noise_mask == 1) | (noise_mask == 2)
        snr_noises, snr_levels, noisy = self._apply_snr_noise(noisy_samples, snr_mask)
        noisy_samples[snr_mask] = noisy

        noise_dict = {
            "snr": snr_noises,
            "rir": rir_samples,
            "noise_mask": noise_mask,
            "snr_levels": snr_levels
        }

        return samples, noisy_samples, noise_dict, *other

    def __len__(self):
        return len(self._data_loader)

    def __iter__(self):
        for idx, batch in enumerate(self._data_loader):
            if self._cache:
                if not self._cache[idx]:
                    self._cache[idx] = self._augment(batch)

                yield self._cache[idx]
            else:
                yield self._augment(batch)
