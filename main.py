import numpy as np
import pyaudio as pa
import struct
import matplotlib.pyplot as plt
import torch


CHUNK = 1024 *2 # Number of samples that will plotted per second in 
FORMAT = pa.paInt16
CHANNELS = 1 
RATE = 8000 # Hz


print("GPU",torch.cuda.is_available())
print("DEVICE",torch.cuda.current_device())
print("NAME",torch.cuda.get_device_name(0))
p = pa.PyAudio()



stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer= CHUNK
)


fig, ax = plt.subplots()
x = np.arange(0,2*CHUNK,2)
line, = ax.plot(x,np.random.rand(CHUNK),'r')
ax.set_ylim(-32000,32000)
ax.set_xlim(0,CHUNK)
fig.show()


while 1:
    data = stream.read(CHUNK)
    dataInt = struct.unpack(str(CHUNK)+'h',data)
    data_np = np.frombuffer(data,dtype='>f4')
    print(data_np,data_np.shape)
    
    
    line.set_ydata(dataInt)
    fig.canvas.draw()
    fig.canvas.flush_events()

    





