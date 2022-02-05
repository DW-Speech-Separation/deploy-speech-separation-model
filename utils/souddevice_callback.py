import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


duration = 30  # seconds
fs = 16000
mixs_ = []


#,type(indata),indata.shape,frames)
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata    
    mixs_.append(np.copy(indata[:, 0]))


with sd.Stream(channels=1, callback=callback,dtype='int16',samplerate=fs):
    sd.sleep(int(duration * 1000))


mixs = np.hstack(mixs_)

print(mixs.shape)

write('resources/wav/realtime/16kHz/yisel_jose_30_16.wav', fs, mixs)#.astype(np.int16))  # Save as WAV file 