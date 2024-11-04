import numpy as np
import torch
import mindspore
from mindnlp.data.io.audio import read
from mindnlp.utils import http_get
from silero_vad_mindspore import load, VADRNNJITMerge


def test_load():
    model = load('silero_vad_v4')
    assert isinstance(model, VADRNNJITMerge)


def test_forward():
    model = load('silero_vad_v4')

    wav_file = http_get('https://models.silero.ai/vad_models/ru.wav', './')
    samples_CT, sample_rate = read(wav_file)

    for i in range(10):
        speech_prob = model(mindspore.Tensor(samples_CT, mindspore.float32), sample_rate)
    print(speech_prob)


def test_audio_forward():
    model = load('silero_vad_v4')

    wav_file = http_get('https://models.silero.ai/vad_models/ru.wav', './')
    samples_CT, sample_rate = read(wav_file)

    speech_prob_batch = model.audio_forward(mindspore.Tensor(samples_CT, mindspore.float32), sample_rate)
    print(speech_prob_batch)

def test_pt_cmp_ms():
    ms_model = load('silero_vad_v4')
    ms_model.eval()
    pt_path = http_get('https://hf-mirror.com/lvyufeng/silero_vad_mindspore/resolve/main/silero_vad_v4.jit', './')
    pt_model = torch.jit.load(pt_path)
    pt_model.eval()
    # wav_file = http_get('https://models.silero.ai/vad_models/ru.wav', './')
    # samples_CT, sample_rate = read(wav_file)
    # print(samples_CT.shape, samples_CT[:10])

    sample_rate = 16000
    for i in range(10):
        samples_CT = np.random.randn(16000)
        speech_prob_ms, _, _ = ms_model(mindspore.Tensor(samples_CT, mindspore.float32), sample_rate)
        speech_prob_pt = pt_model(torch.tensor(samples_CT, dtype=torch.float32), sample_rate)
        print(speech_prob_ms.asnumpy(), speech_prob_pt.detach().numpy())
        assert np.allclose(speech_prob_ms.asnumpy(), speech_prob_pt.detach().numpy(), 1e-3, 1e-3)
