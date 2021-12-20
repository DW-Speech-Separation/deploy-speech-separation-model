import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


duration = 30  # seconds
fs = 8000
mixs_ = []


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)#,type(indata),indata.shape,frames)
    #outdata[:] = indata
    mixs_.append(np.copy(indata[:, 0]))

with sd.Stream(channels=1, callback=callback,samplerate=fs):
    sd.sleep(int(duration * 1000))


mixs = np.hstack(mixs_)
write('resources/wav/realtime/test_padre_2.wav', fs, mixs)  # Save as WAV file 