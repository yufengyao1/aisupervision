import time
import numpy as np
import jax.numpy as jnp
from pydub import AudioSegment
from whisper_jax import FlaxWhisperPipline

# pipeline = FlaxWhisperPipline("openai/whisper-small")
# pipeline = FlaxWhisperPipline("openai/whisper-large-v2",dtype=jnp.float32)
folder="/Users/lingoace/Documents/Repository/TranslaterWeb/weights/whisper-large-v2"
pipeline = FlaxWhisperPipline(folder,dtype=jnp.float32)
print('模型加载完成')
mp4_file="/Users/lingoace/Desktop/mp4/1.wav"

audio = AudioSegment.from_file(mp4_file)  # 课堂音频
audio = audio.set_frame_rate(16000)  # 没有会出问题
audio = audio.set_channels(1)  # 没有会出问题
audio = audio.set_sample_width(2)

waveform=np.array(audio.get_array_of_samples()).astype(np.float32)/32767
input = {"array": waveform, "sampling_rate": 16000,"language":"Chinese"}
print('开始识别')
for i in range(10):
    t1=time.time()
    rr = pipeline(input,return_timestamps=True)
    t2=time.time()
    print(rr)
    print(t2-t1)
        