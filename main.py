import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
from asteroid import ConvTasNet
import contextlib
import queue
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.simpledialog import Dialog
import numpy as np
import sounddevice as sd
import soundfile as sf
import torchaudio.transforms as T
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch.nn.functional as F

def calculate_similarity(speech_embedding,waveform_1,waveform_2,num_layers=3):

        features_1, _ = speech_embedding.extract_features(torch.from_numpy(waveform_1),num_layers)
        features_2, _ = speech_embedding.extract_features(torch.from_numpy(waveform_2),num_layers)        


        print("FEA", features_1[0].shape, features_2[0].shape) 
        # features_1/2 => torch.Size([3, 749, 768])  [bach_size, frames, embedding_size]
        # distance => torch.Size([3, 749]) [bach_size, frames]
        distance =  F.cosine_similarity(features_1[0], features_2[0], dim=1) 
        
        #print("Distance", distance.shape)
        # Luego hacemos el mean por muestra y luego el mean de todas las muestras.
        distance = torch.mean(distance)

        #print(distance.detach().numpy())
        return distance.detach().numpy()

def similarity_definition(speech_embedding,s1,s2,s1_samples,s2_samples): 
    
    #print(s1.shape,s2.shape,s1_samples.shape,s2_samples.shape)

    s1_s1_samples = calculate_similarity(speech_embedding,s1,s1_samples)
    s1_s2_samples = calculate_similarity(speech_embedding,s1,s2_samples)
    s2_s1_samples = calculate_similarity(speech_embedding,s2,s1_samples)
    s2_s2_samples = calculate_similarity(speech_embedding,s2,s2_samples)

    if(s1_s1_samples >= s1_s2_samples):
        s1_samples = np.column_stack([s1_samples,s1])
        s2_samples = np.column_stack([s2_samples,s2])
        return s1_samples,s2_samples
    else:
        s2_samples = np.column_stack([s2_samples,s1])
        s1_samples = np.column_stack([s1_samples,s2])
        return s1_samples,s2_samples
    
    if(s2_s1_samples >= s2_s2_samples):
        s1_samples = np.column_stack([s1_samples,s2])
        s2_samples = np.column_stack([s2_samples,s1])
        return s1_samples,s2_samples
    else:
        s1_samples = np.column_stack([s1_samples,s1])
        s2_samples = np.column_stack([s2_samples,s2])
        return s1_samples,s2_samples

    return s1_samples,s2_samples

def separation(*, q, speech_embedding, best_model, **soundfile_args):
    """Write data from queue to file until *None* is received."""
    # NB: If you want fine-grained control about the buffering of the file, you
    #     can use Python's open() function (with the "buffering" argument) and
    #     pass the resulting file object to sf.SoundFile().
    #with sf.SoundFile(**soundfile_args) as f:
    s1_samples =  np.zeros((1,1)).astype('float32')
    s2_samples =  np.zeros((1,1)).astype('float32')

    first_frame = True
    with sf.SoundFile(**soundfile_args) as f:
        while True:
            data = q.get()
            if data is None:
                break

            f.write(data)

            # Separation
            data = data.reshape(1,data.shape[0])
            out_wavs_after = best_model.separate(data)
            
            
            s1 = out_wavs_after[0,0,:]
            s2 = out_wavs_after[0,1,:]
            
            shape_s = s1.shape[0]

            s1 = s1.reshape(shape_s,1)
            s2 = s2.reshape(shape_s,1)

            #Determinar si es ruido o no

            #Asignar segmento de audio alguno de los dos vectores s1_samples o s2_samples

            if(first_frame): # Significa que es el primer frame

                print(s1_samples.shape,s1.shape)

                s1_samples = np.row_stack([s1_samples,s1])
                s2_samples = np.row_stack([s2_samples,s2])

                print("Before",s1_samples.shape,s1.shape)
                first_frame = False

                s1_samples = s1_samples.reshape(1,s1_samples.shape[0])
                s2_samples = s2_samples.reshape(1,s2_samples.shape[0])

            else:

                s1_samples, s2_samples= similarity_definition(speech_embedding,s1.reshape(1,s1.shape[0]),
                s2.reshape(1,s2.shape[0]),
                s1_samples,#.reshape(1,s1_samples.shape[0]),
                s2_samples)#.reshape(1,s2_samples.shape[0]))

            len_s = s1_samples.shape[1]
            #Definir orden reproducción

            print("SIZE",s1_samples.shape, s2_samples.shape)
            
            sd.play(np.column_stack([s1_samples[0,len_s-shape_s:],s2_samples[0,len_s-shape_s:]]),samplerate=8000)
            #s1_samples = np.concatenate((s1_samples,s1))
            #s2_samples = np.concatenate((s2_samples,s2))

            
            
    sf.write('s1.wav', s1_samples[1:], 8000)
    sf.write('s2.wav', s2_samples[1:], 8000)


class SettingsWindow(Dialog):
    """Dialog window for choosing sound device."""
    def body(self, master):
        ttk.Label(master, text='Select host API:').pack(anchor='w')
        hostapi_list = ttk.Combobox(master, state='readonly', width=50)
        hostapi_list.pack()
        hostapi_list['values'] = [hostapi['name']
                                  for hostapi in sd.query_hostapis()]

        ttk.Label(master, text='Select sound device:').pack(anchor='w')
        device_ids = []
        device_list = ttk.Combobox(master, state='readonly', width=50)
        device_list.pack()

        def update_device_list(*args):
            hostapi = sd.query_hostapis(hostapi_list.current())
            nonlocal device_ids
            device_ids = [
                idx
                for idx in hostapi['devices']
                if sd.query_devices(idx)['max_output_channels'] > 0]
            device_list['values'] = [
                sd.query_devices(idx)['name'] for idx in device_ids]
            default = hostapi['default_output_device']
            if default >= 0:
                device_list.current(device_ids.index(default))
                device_list.event_generate('<<ComboboxSelected>>')

        def select_device(*args):
            self.result = device_ids[device_list.current()]

        hostapi_list.bind('<<ComboboxSelected>>', update_device_list)
        device_list.bind('<<ComboboxSelected>>', select_device)

        with contextlib.suppress(sd.PortAudioError):
            hostapi_list.current(sd.default.hostapi)
            hostapi_list.event_generate('<<ComboboxSelected>>')


class RecGui(tk.Tk):

    stream = None
    
    def __init__(self):
        super().__init__()

        self.title('Realtime Speech Separation - UdeA - In2Lab')

        padding = 10


        f = ttk.Frame()

        self.rec_button = ttk.Button(f)
        self.rec_button.pack(side='left', padx=padding, pady=padding)

        self.settings_button = ttk.Button(
            f, text='Configuración', command=self.on_settings)
        self.settings_button.pack(side='left', padx=padding, pady=padding)

        f.pack(expand=True, padx=padding, pady=padding)

        self.file_label = ttk.Label(text='<file name>')
        self.file_label.pack(anchor='w')

        self.input_overflows = 0
        self.status_label = ttk.Label()
        self.status_label.pack(anchor='w')

        self.meter = ttk.Progressbar()
        self.meter['orient'] = 'horizontal'
        self.meter['mode'] = 'determinate'
        self.meter['maximum'] = 1.0
        self.meter.pack(fill='x')

        # We try to open a stream with default settings first, if that doesn't
        # work, the user can manually change the device(s)
        self.create_stream()

        self.recording = self.previously_recording = False
        self.audio_q = queue.Queue()
        self.peak = 0
        self.metering_q = queue.Queue(maxsize=1)

        self.data_frame = np.zeros((1,1)).astype('float32')
        self.N_SAMPLES = 8000

        self.protocol('WM_DELETE_WINDOW', self.close_window)
        self.init_buttons()
        self.update_gui()

        # 1. Load model
        self.best_model = self.load_model()
        self.speech_embedding = self.speech_embedding_initialize()

    def speech_embedding_initialize(self):
        print("Downloading Speech Embedding.....") #force_download=True
        original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").cuda()
        speech_embedding = import_huggingface_model(original)
        return speech_embedding

    def load_model(self):
        print("1.....Cargando modelo.....")
        path_best_model = "checkpoint/best_model_CF_100.pth"
        best_model  = ConvTasNet.from_pretrained(path_best_model)
        best_model.cuda()

        model_device = next(best_model.parameters()).device
        print("Modelo cargado ...  en ",model_device)
        return best_model

    def create_stream(self, device=None):
        if self.stream is not None:
            self.stream.close()
        self.stream = sd.InputStream(
            device=device, channels=1, samplerate=8000,dtype='float32',callback=self.audio_callback)

        
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status.input_overflow:
            # NB: This increment operation is not atomic, but this doesn't
            #     matter since no other thread is writing to the attribute.
            self.input_overflows += 1
        # NB: self.recording is accessed from different threads.
        #     This is safe because here we are only accessing it once (with a
        #     single bytecode instruction).
        if self.recording:

            self.previously_recording = True        
            self.data_frame = np.concatenate((self.data_frame,indata.copy()))

            if(self.data_frame.shape[0] >= self.N_SAMPLES):
                #Enviar a colar de separación                 
                 self.audio_q.put(self.data_frame[1:,:])
                # Reinicio frame
                 self.data_frame = np.zeros((1,1)).astype('float32')
        else:
            if self.previously_recording:
                self.audio_q.put(None)
                self.previously_recording = False

        self.peak = max(self.peak, np.max(np.abs(indata)))
        try:
            self.metering_q.put_nowait(self.peak)
        except queue.Full:
            pass
        else:
            self.peak = 0

    def on_rec(self):
        self.settings_button['state'] = 'disabled'
        self.recording = True

        filename = tempfile.mktemp(
            prefix='delme_rec_gui_', suffix='.wav', dir='')

        if self.audio_q.qsize() != 0:
            print('WARNING: Queue not empty!')
        self.thread = threading.Thread(
            target=separation,
            kwargs=dict(
                file=filename,
                mode='x',
                samplerate=int(self.stream.samplerate),
                channels=self.stream.channels,
                q=self.audio_q,
                best_model = self.best_model,
                speech_embedding = self.speech_embedding
            ),
        )
        self.thread.start()

        # NB: File creation might fail!  For brevity, we don't check for this.

        self.rec_button['text'] = 'stop'
        self.rec_button['command'] = self.on_stop
        self.rec_button['state'] = 'normal'
        self.file_label['text'] = filename

    def on_stop(self, *args):
        self.rec_button['state'] = 'disabled'
        self.recording = False
        self.wait_for_thread()

    def wait_for_thread(self):
        # NB: Waiting time could be calculated based on stream.latency
        self.after(10, self._wait_for_thread)

    def _wait_for_thread(self):
        if self.thread.is_alive():
            self.wait_for_thread()
            return
        self.thread.join()
        self.init_buttons()

    def on_settings(self, *args):
        w = SettingsWindow(self, 'Configuraciones')
        self.create_stream(device=w.result)

    def init_buttons(self):
        self.rec_button['text'] = 'Grabar'
        self.rec_button['command'] = self.on_rec
        if self.stream:
            self.rec_button['state'] = 'normal'
        self.settings_button['state'] = 'normal'

    def update_gui(self):
        self.status_label['text'] = 'input overflows: {}'.format(
            self.input_overflows)
        try:
            peak = self.metering_q.get_nowait()
        except queue.Empty:
            pass
        else:
            self.meter['value'] = peak
        self.after(100, self.update_gui)

    def close_window(self):
        if self.recording:
            self.on_stop()
        self.destroy()

def main():
    app = RecGui()

    app.geometry("500x400")
    app.mainloop()


if __name__ == '__main__':
    main()