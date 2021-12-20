import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
from asteroid import ConvTasNet
import soundfile as sf
import io
import queue
import threading

def load_model():
    print("1.....Cargando modelo.....")
    path_best_model = "checkpoint/best_model_CF_100.pth"
    best_model  = ConvTasNet.from_pretrained(path_best_model)
    best_model.cuda()

    model_device = next(best_model.parameters()).device
    print("Modelo cargado ...  en ",model_device)
    return best_model


def separation(mixture):
    #print("MIXTURE =>", mixture.shape)
    #print("4....Separando.....")

    out_wavs_after = best_model.separate(mixture)
    #print("SEPA =>"+str(i),out_wavs_after.shape)
    s1 = out_wavs_after[0,0,:]
    s2 = out_wavs_after[0,1,:]

    #print(s1.shape,s2.shape)

    return s1,s2


mixs_ = []
s1_sources = []
s2_sources = []
fs = 8000  # Sample rate
duration = 7  # seconds
q = queue.Queue() # Cola donde almacenan los datos.
recording = True
blocksize = duration*fs #Número de muestras a capturar en tiempo real.


name ="test_paola"
filename  = "resources/wav/realtime/"+name+".wav"
buffersize = 70
i = 0


# 1. Load model
best_model = load_model()
print("3....Grabando.....")



def callback(indata,frames, time, status):
    global q,i,recording
    if status:
        print(status, type(indata),indata.shape)
    
    q.put(np.copy(indata[:, 0]))
    recording = True
    #outdata[:] = indata
    #write('resources/wav/realtime/seconds/test_'+str(i)+'.wav', fs, indata)  # Save as WAV file 
    i = i+1


def full_queue():
    with sf.SoundFile(filename) as f:
        for _ in range(buffersize):
            data = f.read(blocksize, dtype='float32',always_2d=True)
            print(data.shape)
            if not len(data):
                break
            q.put_nowait(data[:,0])  # Pre-fill queue
        


full_queue()
#print("llenando cola", q.qsize())


def update_separation():
    global recording,mixs_,s1_sources,s2_sources
    while q.qsize() > 0:
        
        try:
            input = q.get_nowait()
            mixture = input.reshape(1, input.shape[0])
            s1,s2 = separation(torch.from_numpy(mixture).to('cuda'))    
            mixs_.append(input)
            s1_sources.append(s1)
            s2_sources.append(s2)
            
        except queue.Empty:
            continue


#th = threading.Thread(target=update_separation)
#th.start()

update_separation()

"""
with sd.InputStream(channels=1, callback=callback,samplerate=fs,dtype='float32', blocksize=blocksize):
    th.start()
    sd.sleep(int(duration * 1000))
    recording = False
    print("SIZE COLA", q.qsize(),i)
"""


mixs = np.hstack(mixs_)
s1_sources = np.hstack([s.cpu() for s in s1_sources])
s2_sources = np.hstack([s.cpu() for s in s2_sources])


name = name +"_"+str(duration)+"S"

write('resources/wav/realtime/'+name+'.wav', fs, mixs)  # Save as WAV file 
write('resources/separations/realtime/'+name+'_s1.wav',fs,s1_sources)
write('resources/separations/realtime/'+name+'_s2.wav',fs, s2_sources)

