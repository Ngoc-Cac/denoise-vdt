import torch

from torchcodec.decoders import AudioDecoder

from typing import Literal
from ...logger import get_logger

logger = get_logger(__name__)


class PreprocessingService:
    def __init__(self):
        model, (*_, vad_iter, _) = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self._vad_iter = vad_iter(model)

        logger.debug("Initialised preprocessing service")

    def load_file(
        self,
        file: str,
        num_channels: Literal[1, 2] = 1,
        max_dur: int | float = 10,
        stream: bool = True
    ):
        sr=16000  # must be 16k or 8k
        decoder = AudioDecoder(
            file,
            sample_rate=sr,
            num_channels=num_channels
        )
        if not stream:
            return decoder.get_all_samples().data

        total_frames = sr * decoder.metadata.duration_seconds
        win_len, cur = 512, 0
        yield_thres = (sr * max_dur) // win_len
        audio_buffer = []
        while cur < total_frames:
            # 512 samples in 16k Hz
            next = min(cur + win_len, total_frames)
            samples = decoder.get_samples_played_in_range(
                cur / sr,
                next / sr
            ).data
            cur = next

            if samples.shape[-1] != win_len:
                continue

            vad_res = self._vad_iter(samples)
            if vad_res and 'end' in vad_res and len(samples) >= yield_thres:
                chunk = torch.concat(audio_buffer, dim=-1)
                audio_buffer = []
                self._vad_iter.reset_states()

                yield chunk, sr
            else:
                audio_buffer.append(samples)

        if len(audio_buffer):
            yield torch.concat(audio_buffer, dim=-1), sr

        self._vad_iter.reset_states()
