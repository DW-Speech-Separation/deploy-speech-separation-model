import torchaudio.functional as F
import torchaudio
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from IPython.display import Audio, display

def plot_spectrogram(audio,audio_name,sr = 8000, hop_length = 512, win_length = 1024, y_axis ='mel',windows_type='hamming'):
  """
  Nuestro eje y corresponde a frecuencias espaciadas linealmente
  producidas por la transformada discreta de Fourier
  """  

  if (y_axis == 'mel'):
    M = librosa.feature.melspectrogram(y=audio, sr=sr,hop_length=hop_length,win_length=win_length,window=windows_type)
    S_db = librosa.power_to_db(M, ref=np.max)    
    title_ = 'Mel-Frequency Spectrogram audio '
  else:
    D = librosa.stft(audio,win_length=win_length,hop_length=hop_length, window=windows_type)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
  
  
  fig, ax = plt.subplots()
  img = librosa.display.specshow(S_db, x_axis='time', y_axis=y_axis, ax=ax, sr=sr, hop_length=hop_length)
  
  if (y_axis == 'linear'):
    title_ = 'Linear-frequency power spectrogram - audio '
  if (y_axis == 'log'):
    title_ = 'Log-frequency power spectrogram - audio '

  ax.set(title=title_+ audio_name)
  fig.colorbar(img, ax=ax, format="%+2.f dB")
  
  return ax


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show()

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_speech_sample(SAMPLE_WAV_SPEECH_PATH, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")


file_name  ="test_pacho"
SAMPLE_WAV_SPEECH_PATH ="resources/wav/realtime/"+file_name+".wav"

files_wave = ["Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042"]#"test_pacho","test_padre_1_noise","test_carlos","test_madre_2_reduce_noise"]

for f in files_wave:
    SAMPLE_WAV_SPEECH_PATH ="resources/wav/realtime/"+f+".wav"
    waveform, sample_rate = get_speech_sample(SAMPLE_WAV_SPEECH_PATH,8000)
    print("Original")
    name = "Original "+f
    mix_spec = plot_spectrogram(waveform.numpy()[0,:],name)
    mix_spec.figure.savefig("/home/josearangos/Documentos/UdeA/TG-Separación-Fuentes/Simulacion_canal_telefonico/"+name+".png",bbox_inches='tight', pad_inches=0)
    
    plot_specgram(waveform, sample_rate, name)
    
    print("Codec")
    
    
    configs = [
    ({"format": "gsm"}, "GSM-FR")]
    
    #for param, title in configs:
    #    augmented = F.apply_codec(waveform, sample_rate, **param)

    augmented = waveform

    #name = "GSM-FR "+f

    name = "RESAMPLE_"+f
    print(augmented.shape)
    mix_spec = plot_spectrogram(augmented.numpy()[0,:],name)
    mix_spec.figure.savefig("/home/josearangos/Documentos/UdeA/TG-Separación-Fuentes/Simulacion_canal_telefonico/"+name+".png",bbox_inches='tight', pad_inches=0)
    
    plot_specgram(augmented, sample_rate, name)
    torchaudio.save("/home/josearangos/Documentos/UdeA/TG-Separación-Fuentes/Simulacion_canal_telefonico/"+name+".wav",augmented,sample_rate,format="wav")