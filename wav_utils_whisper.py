import os
import time
import math
import whisper
import traceback
import numpy as np
from zhconv import convert
from fbank import logfbank
from pydub import AudioSegment
from textana.txt_analyse import TxtAna
from pydub.silence import detect_nonsilent

class ASR:
    def __init__(self):
        self.model = whisper.load_model("small",download_root=os.path.join(os.path.dirname(__file__), 'weights/whisper'))
        self.txtAna = TxtAna()

    def recognize_wav(self, mp4_file, language, role, room_id):    
        try:
            audio = AudioSegment.from_file(mp4_file, "mp4")  # 课堂音频
            result = []
            audio = audio.set_frame_rate(16000)  # 没有会出问题
            audio = audio.set_channels(1)  # 没有会出问题
            audio = audio.set_sample_width(2)

            waveform=np.array(audio.get_array_of_samples()).astype(np.float32)/32767

            # waveform=waveform[:10181632]

            r = self.model.transcribe(waveform,language='Chinese') #整段翻译

            segments=r["segments"]
            for item in segments:
                result.append([int(1000*item["start"]),int(1000*item["end"]),item["text"]])
                print([int(1000*item["start"]),int(1000*item["end"]),item["text"]])

            # chunks_origin = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-75, seek_step=20)  # 结果数组起止点单位为300ms

            word_num = 0  # 字数
            speak_time = 0  # 开口时长

            speak_times=len(result)
            praise_times = 0  # 夸赞次数
            valid_time = 0  # 有效开口时长，去除少于2个字的句子
            valid_num = 0  # 有效词个数，去除少于1个字的句子
            valid_word_count = 5  # 讲1个字以上的算入语速
            talk_with_parent_detail = ""
            talk_about_homework_detail = ""
            talk_with_parent = '否'
            talk_about_homework = '否'

            if language == "cn":
                for item in result:
                    word_num += len(item[2])
                    if len(item[2]) > 0:
                        speak_time += (item[1]-item[0])
                    current_speed = 1000*len(item[2])/(item[1]-item[0])  # 单句纯语速
                    if len(item[2]) > valid_word_count and current_speed < 10 and current_speed > 1:
                        valid_num += len(item[2])
                        valid_time += (item[1]-item[0]+800)
                    if role == 0:  # 识别夸赞次数
                        is_praise = self.txtAna.is_praise(item[2])
                        if is_praise:
                            praise_times += 1
                        # 识别跟家长沟通
                        r = self.txtAna.talk_about_homework(item[2])
                        if r:
                            talk_about_homework = '是'
                            talk_about_homework_detail+=('{0}:{1},'.format(math.floor(item[0]/60000), int(item[0]/1000) % 60))

                        r = self.txtAna.talk_with_parent(item[2])
                        if r:
                            talk_with_parent = '是'
                            talk_with_parent_detail+=('{0}:{1},'.format(math.floor(item[0]/60000), int(item[0]/1000) % 60))
            else:
                current_word_num = item[2].count('▁')
                if current_word_num > 0:
                    speak_time += (item[1]-item[0])
                word_num += current_word_num
                current_speed = 1000*current_word_num/(item[1]-item[0])  # 单句纯语速
                if item[2].count('▁') > valid_word_count and current_speed < 10 and current_speed > 1:
                    valid_num += item[2].count('▁')
                    valid_time += (item[1]-item[0]+800)

            speak_speed = 0 if valid_time == 0 else round(1000*valid_num/valid_time, 2)  # 语速
            
            return speak_time, speak_times, speak_speed, praise_times, word_num, result, result, talk_with_parent, talk_about_homework, talk_with_parent_detail, talk_about_homework_detail
        except Exception as ex:
            traceback.print_exc()
            print("{0}音频分析出错，可能缺少音频文件！{1}".format(room_id,str(ex)))
            return 0, 0, 0, 0, 0, [], [], '', '', [], []
