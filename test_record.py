import numpy as np
from scipy.io import wavfile
import pyaudio
import soundfile as sf


def sound(array, fs=8000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


def record(duration=30, fs=8000):
    nsamples = duration*fs
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True,
                    frames_per_buffer=nsamples)

                    
    buffer = stream.read(nsamples)
    array = np.frombuffer(buffer, dtype='int16')
    stream.stop_stream()
    stream.close()
    p.terminate()
    return array


#sound(data, fs=4000)
my_recording = record()
print("Example",type(my_recording),my_recording.shape)
sound(my_recording)



RATE = 8000 # Hz


file_name = "gonorrea_yuli.wav"
local_save_dir = "resources/wav/"


sf.write(local_save_dir + file_name, my_recording, RATE)