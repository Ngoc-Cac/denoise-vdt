import pandas as pd, datasets
import torch

from torch.utils.data import DataLoader
from torchaudio import functional as F
from transformers import AutoModel
from safetensors import load_file

from . import _ROOT
from .datasets.data_loader import NoiseAugmentLoader
from .datasets.datasets import (
    MSSNSDataset,
    FLAIRDataset,
    SpeechTranscriptDataset
)


_SOURCES = [
    ("fosd", "doof-ferb/fpt_fosd"),
    ("vlsp", "doof-ferb/vlsp2020_vinai_100h")
]


def load_vn_speech_transcript_datasets(
    target_sr,
    train_ratio,
    snr_levels: torch.Tensor | None = None,
    return_data_loader: bool = False,
    *,
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    rir_test_ratio: float = .4,
    mos_bak_thres: int | float = 4,
    mos_sig_thres: int | float = 3.4,
    seed=None
):
    if snr_levels is None:
        snr_levels = torch.linspace(0, 20, 21)

    data = []
    for name, path in _SOURCES:
        df = pd.read_csv(f"{_ROOT}/data/dnsmos/{name}.csv", index_col=0)
        mask = (df['mos_bak'] > mos_bak_thres) & (df['mos_sig'] > mos_sig_thres)

        data.append(datasets.load_dataset(path, split='train').select(
            df.loc[mask].index.to_list()
        ))
        print(f"Kept {mask.sum()} rows from {name}")

    speech_splits = datasets.concatenate_datasets(data).train_test_split(
        train_size=train_ratio,
        seed=seed
    )

    preprocessor = lambda x, sr: F.resample(
        x, sr, target_sr
    )

    snr_train = MSSNSDataset(snr_levels, preprocessor=preprocessor, split='train')
    snr_test = MSSNSDataset(snr_levels, preprocessor=preprocessor, split='test')
    rir_dataset = FLAIRDataset(preprocessor=preprocessor)
    rir_splits = rir_dataset.train_test_split(rir_test_ratio, seed=seed)

    train_dataset = SpeechTranscriptDataset(
        dataset=speech_splits['train'],
        preprocessor=preprocessor
    )
    test_dataset = SpeechTranscriptDataset(
        dataset=speech_splits['test'],
        preprocessor=preprocessor
    )

    if return_data_loader:
        train_loader = NoiseAugmentLoader(
            DataLoader(
                train_dataset,
                train_batch_size,
                collate_fn=train_dataset.collate_fn
            ),
            snr_dataset=snr_train,
            rir_dataset=rir_splits['train']
        )
        test_loader = NoiseAugmentLoader(
            DataLoader(
                test_dataset,
                test_batch_size,
                collate_fn=test_dataset.collate_fn
            ),
            snr_dataset=snr_test,
            rir_dataset=rir_splits['test'],
            resample=False
        )
    else:
        train_loader, test_loader = None, None

    return {
        "train": {
            "noise_dataset": {"snr": snr_train, "rir": rir_splits['train']},
            "speech_dataset": train_dataset,
            "loader": train_loader
        },
        "test": {
            "noise_dataset": {"snr": snr_test, "rir": rir_splits['test']},
            "speech_dataset": test_dataset,
            "loader": test_loader
        }
    }


def load_dasheng(
    denoiser_ckpt: str | None = None,
    freeze_for_finetune: bool = True
):
    model = AutoModel.from_pretrained(
        "mispeech/dasheng-denoiser",
        trust_remote_code=True
    )

    if freeze_for_finetune:
        freeze_layers = [model.feature_extractor, model.backbone, model.head]
        for layer in freeze_layers:
            layer.requires_grad_(False)

    if load_from_checkpoint:
        model.second_encoder.load_state_dict(load_file(denoiser_ckpt))

    return model
