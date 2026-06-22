import torch

from torchaudio import functional as F
from chunkformer import ChunkFormerModel
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    pipeline
)


def _load_wav2vec2vn(path, device):
    processor = Wav2Vec2Processor.from_pretrained(path)
    model = Wav2Vec2ForCTC.from_pretrained(path).eval().to(device)

    def transcribe(wav):
        input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        logits = model(input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(pred_ids)[0]
    return transcribe


def load_wav2ve2_160_model(device):
    return _load_wav2vec2vn("khanhld/wav2vec2-base-vietnamese-160h", device)


def load_wav2vec2_250_model(device):
    return _load_wav2vec2vn("nguyenvulebinh/wav2vec2-base-vietnamese-250h", device)


def load_PhoWhisper_model(device):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="vinai/PhoWhisper-small",
        device=device,
    )

    def transcribe(wav):
        return pipe(wav)['text']
    return transcribe


def load_chunkformer(device):
    model = ChunkFormerModel.from_pretrained(
        "khanhld/chunkformer-ctc-large-vie"
    ).to(device)

    def transcribe(audio_file):
        return model.endless_decode(
            audio_path=audio_file,
            # chunk_size=64,
            # left_context_size=128,
            # right_context_size=128,
            # total_batch_duration=14400,  # in seconds
            return_timestamps=False
        )
    return transcribe
