import numpy as np
from silero_vad_mindspore import load, FixedVADIterator

def test_vad_iterator():
    model = load('silero_vad_v4')
    vac = FixedVADIterator(model)
    #   vac = VADIterator(model)  # the second case crashes with this

    # this works: for both
    audio_buffer = np.array([0]*(512),dtype=np.float32)
    out = vac(audio_buffer)
    print(out)
    # this crashes on the non FixedVADIterator with 
    # ops.prim.RaiseException("Input audio chunk is too short", "builtins.ValueError")
    audio_buffer = np.array([0]*(512-1),dtype=np.float32)
    out = vac(audio_buffer)
    print(out)
