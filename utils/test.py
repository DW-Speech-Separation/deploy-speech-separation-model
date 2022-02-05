import torch
from asteroid import ConvTasNet
import soundfile as sf
import timeit

# 1 Global variable


file_name = "yisel_jose_1_8"
PATH_TEST ="resources/wav/realtime/"+file_name+".wav"
pre =""

#PATH_TEST = "resources/wav/realtime/test.wav"
#file_name = "test"

local_save_dir = "resources/separations/16kHz/"

# 2. Load model
path_best_model = "checkpoint/best_model_CF_100.pth"
best_model  = ConvTasNet.from_pretrained(path_best_model)
best_model.cuda()

# 3. read and prepro mix
mixture, _ = sf.read(PATH_TEST, dtype="float32", always_2d=True)
print(mixture.shape)
mixture = mixture[:,0]
mixture = mixture.reshape(1,mixture.shape[0])


print("SHAPEEEE", mixture.shape)
# 4. Separation speakers
out_wavs_after = best_model.separate(mixture)


# 5. Save results
rate=8000
s1 = out_wavs_after[0,0,:]
s2 = out_wavs_after[0,1,:]


sf.write(local_save_dir + pre +file_name+"_s1.wav", s1, rate)
sf.write(local_save_dir + pre + file_name+"_s2.wav", s2, rate)

print("GUARDO", local_save_dir+file_name)