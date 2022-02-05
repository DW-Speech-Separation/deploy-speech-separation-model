import numpy as np
from numpy.lib.npyio import NpzFile
import pyaudio as pa
import struct
import matplotlib.pyplot as plt
import torch
from asteroid import ConvTasNet
import soundfile as sf
import io
from numpy import savetxt


import wave

POINTS = 8000

RECORD_SECONDS = 10
CHUNK = 1024*1 #POINTS*3 # Number of samples that will plotted per second in 
FORMAT = pa.paFloat32
CHANNELS = 1 
RATE = 8000 # Hz
file_name = "test_1"
local_save_dir = "resources/separations/"

frames = []
i = 0

s1_sources = []
s2_sources = []





print("GPU",torch.cuda.is_available())
print("DEVICE",torch.cuda.current_device())
print("NAME",torch.cuda.get_device_name(0))
p = pa.PyAudio()



stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    #output=True,
    frames_per_buffer= CHUNK
)


def load_model():
    print("1.....Cargando modelo.....")
    path_best_model = "checkpoint/best_model.pth"
    best_model  = ConvTasNet.from_pretrained(path_best_model)
    best_model.cuda()

    model_device = next(best_model.parameters()).device
    print("Modelo cargado ...  en ",model_device)
    return best_model


def save_wave_file(filename,frames):
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()



# 1. Load model
best_model = load_model()


print("3....Grabando.....")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    #frames.append(np.array(struct.unpack(str(CHUNK*4)+'B',data),dtype='b'))
    frames.append(np.fromstring(data,'float32'))

np_frames = np.hstack(frames)

print("TYPE",type(np_frames),np_frames.shape)
mixture = np_frames.reshape(1, np_frames.shape[0])


print("MIXTURE =>"+str(i), mixture.shape)
print("4....Separando.....")

"""
out_wavs_after = best_model.separate(mixture)
#print("SEPA =>"+str(i),out_wavs_after.shape)
s1 = out_wavs_after[0,0,:]
s2 = out_wavs_after[0,1,:]


s1_sources.append(s1)
s2_sources.append(s2)
"""


# stop Recording
print("4....Stop......")
stream.stop_stream()
stream.close()
p.terminate()



print("6....Guardando...")
save_wave_file('resources/wav/realtime/test.wav',np_frames)
#save_wave_file('resources/separations/realtime/test_s1.wav',s1_sources)
#save_wave_file('resources/separations/realtime/test_s2.wav',s2_sources)