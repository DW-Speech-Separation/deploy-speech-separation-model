import torchaudio.functional as F
import torchaudio
import torch
import matplotlib.pyplot as plt
import math



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
  plt.show(block=False)


configs = [
    #({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
    ({"format": "gsm"}, "GSM-FR"),
    #({"format": "mp3", "compression": -9}, "MP3"),
    #({"format": "vorbis", "compression": -1}, "Vorbis"),
]

file_name  ="test_padre_1_noise"
PATH_TEST ="resources/wav/realtime/"+file_name+".wav"
waveform, sample_rate = torchaudio.load(PATH_TEST)

"""
plot_specgram(waveform, sample_rate, title="Original")
#play_audio(waveform, sample_rate)
for param, title in configs:
  augmented = F.apply_codec(waveform, sample_rate, **param)
  plot_specgram(augmented, sample_rate, title=title)
  #play_audio(augmented, sample_rate)
  print(augmented.shape, "Codec")
  path = file_name+"_codec_GSM.wav"
  torchaudio.save(path, augmented, sample_rate)
"""


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


def get_rir_sample(*, resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  rir = rir / torch.norm(rir, p=2)
  rir = torch.flip(rir, [1])
  return rir, sample_rate

def get_speech_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)

def get_noise_sample(*, resample=None):
  return _get_sample(SAMPLE_NOISE_PATH, resample=resample)


SAMPLE_NOISE_PATH =  "bg.wav"
SAMPLE_WAV_SPEECH_PATH = PATH_TEST# "speech.wav"
SAMPLE_RIR_PATH = "rir.wav"



sample_rate = 8000
speech, _ = get_speech_sample(resample=sample_rate)

# Apply RIR
"""
rir, _ = get_rir_sample(resample=sample_rate, processed=True)
speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
speech = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
"""

"""
noise, _ = get_noise_sample(resample=sample_rate)
noise = noise[:, :speech.shape[1]]
snr_db = 8
scale = math.exp(snr_db / 10) * noise.norm(p=2) / speech.norm(p=2)
speech = (scale * speech + noise) / 2
"""

"""
# Apply filtering and change sample rate
speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
  speech,
  sample_rate,
  effects=[
      ["lowpass", "3000"],
      ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
      ["rate", "8000"],
  ],
)

# Apply telephony codec
speech = F.apply_codec(speech, sample_rate, format="gsm")

path =file_name+"_codec_GSM.wav"
torchaudio.save(path, speech, sample_rate)


plot_specgram(speech, sample_rate, title="GSM Codec Applied")
plt.show()
"""