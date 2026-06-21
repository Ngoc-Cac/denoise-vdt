import torch
import torch.nn as nn

from transformers import Trainer, PreTrainedModel

from .datasets import NoiseAugmentLoader


class DenoiserTrainer(Trainer):
    def __init__(
        self,
        audio_encoder: PreTrainedModel,
        train_loader: NoiseAugmentLoader,
        eval_loader: NoiseAugmentLoader,
        loss_fn = nn.MSELoss(),
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._train_noise_loader = train_loader
        self._eval_noise_loader = eval_loader
        self._loss_fn = loss_fn

        self._audio_encoder = audio_encoder
        self._audio_encoder.requires_grad_(False)

    def get_train_dataloader(self):
        return self._train_noise_loader

    def get_eval_dataloader(self, eval_dataset=None):
        return self._eval_noise_loader

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None
    ):
        clean, noisy = inputs[0], inputs[1]

        with torch.no_grad():
            # The denoiser actually returns (B, hidden, C) not (B, C, hidden)
            clean_embeds = self._audio_encoder(clean).transpose(1, 2).contiguous()
            noisy_embeds = self._audio_encoder(noisy)

        denoised_embeds = model(noisy_embeds)
        loss = self._loss_fn(denoised_embeds, clean_embeds)

        return (loss, denoised_embeds) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)

        with torch.no_grad():
            res = self.compute_loss(model, inputs, return_outputs=True)
        return (res[0], None, None) if prediction_loss_only else (*res, None)
