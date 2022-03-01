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


data, fs = sf.read("yis_1.wav", always_2d=True)

print(data.shape)

def audio_callback(indata, frames, time, status):
        print(indata.shape)


stream= sd.OutputStream(samplerate=fs, dtype='float32', channels=1,callback=audio_callback ,blocksize=8000)
stream.start()
