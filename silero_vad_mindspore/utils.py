
# This is copied from silero-vad's vad_utils.py:
# https://github.com/snakers4/silero-vad/blob/f6b1294cb27590fb2452899df98fb234dfef1134/utils_vad.py#L340

# Their licence is MIT, same as ours: https://github.com/snakers4/silero-vad/blob/f6b1294cb27590fb2452899df98fb234dfef1134/LICENSE
import mindspore
import numpy as np

from mindnlp.core import ops
from mindnlp.utils.download import http_get
from mindnlp.core.serialization import safe_load_file

from .silero_vad_v4 import VADRNNJITMerge

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not ops.is_tensor(x):
            try:
                x = mindspore.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate)
        speech_prob = speech_prob.item()
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None

#######################
# this is our workaround for Silero v5 requiring at least 512-sized audio chunks 
# (see https://github.com/ufal/whisper_streaming/issues/116 )

class FixedVADIterator(VADIterator):

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([],dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x) 
        if len(self.buffer) >= 512:
            ret = super().__call__(self.buffer, return_seconds=return_seconds)
            self.buffer = np.array([],dtype=np.float32)
            return ret
        return None

def load(model, save_dir='./'):
    model_list = ['silero_vad_v4']
    if model not in model_list:
        raise ValueError(f'only support {model_list}, but got `{model}`.')
    url = f"https://hf-mirror.com/lvyufeng/silero_vad_mindspore/resolve/main/{model}.safetensors"
    file = http_get(url, save_dir)
    state_dict = safe_load_file(file)
    
    model = VADRNNJITMerge()
    model.load_state_dict(state_dict)
    model.eval()

    return model
