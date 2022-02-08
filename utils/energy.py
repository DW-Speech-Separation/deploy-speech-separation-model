import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd



"""
Energy

https://musicinformationretrieval.com/energy.html
"""

data, fs = sf.read('utils/mix_test_energy.wav')

def get_max_energy(audio, segment_size=8000):
   energies_audio = []
   blocks = []
   for i in range(0, len(audio), segment_size):
       block = audio[i:i+segment_size]
       energies_audio.append(np.sum(np.abs(block**2)))
       blocks.append(block)
   
   energies_audio = np.array(energies_audio)
   index_max_energy = np.argmax(energies_audio)
   max_energy_block = blocks[index_max_energy]
   max_energy = energies_audio[index_max_energy]
   return max_energy,max_energy_block

max_energy,max_energy_block = get_max_energy(data, segment_size=8000)
sd.play(max_energy_block,samplerate=8000)
sd.wait()
print(max_energy)




"""

rms = [np.sqrt(np.mean(block**2)) for block in                 
       sf.blocks('utils/mix_test_energy.wav', blocksize=8000)] #overlap=512

print(rms)

i = 0
blocks = []
for block in  sf.blocks('utils/mix_test_energy.wav', blocksize=8000):
    
    rms = np.sum(np.abs(block**2))
    plt.title("RMS "+str(rms))
    plt.plot(block)
    plt.savefig("Segmente "+str(i)+" RMS "+str(rms)+".png")
    plt.show()
    i = i +1
    blocks.append(block)

sd.play(blocks[8],samplerate=8000)
sd.wait()

"""
