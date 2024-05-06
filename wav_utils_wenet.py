import os
import gc
import math
import time
import copy
import array
import traceback
import numpy as np
import onnxruntime
from fbank import logfbank
from pydub import AudioSegment
from textana.txt_analyse import TxtAna
from pydub.silence import detect_nonsilent
from swig_decoders import map_batch, ctc_beam_search_decoder_batch, TrieVector, PathTrie


class ASR:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4  # 设置线程数
        opts.inter_op_num_threads = 4  # 设置parallel线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session_encoder = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet', 'encoder.onnx'),
                                                                 opts, providers=provider)
        self.onnx_session_decoder = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet', 'decoder.onnx'),
                                                                 opts, providers=provider)

        # self.onnx_session_encoder_en = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet_en', 'encoder.onnx'),
        #                                                             opts, providers=provider)
        # self.onnx_session_decoder_en = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet_en', 'decoder.onnx'),
        #                                                             opts, providers=provider)
        self.vocabulary = []
        self.char_dict = {}
        self.vocabulary_en = []
        self.char_dict_en = {}
        with open(os.path.join(os.path.dirname(__file__), 'weights/wenet', 'words.txt'), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                self.char_dict[int(arr[1])] = arr[0]
                self.vocabulary.append(arr[0])
        with open(os.path.join(os.path.dirname(__file__), 'weights/wenet_en', 'words.txt'), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                self.char_dict_en[int(arr[1])] = arr[0]
                self.vocabulary_en.append(arr[0])
        self.eos = len(self.char_dict) - 1
        self.sos = self.eos
        self.mode = "attention_rescoring"
        self.num_processes = 4  # decoders线程数
        self.fp16 = False
        self.IGNORE_ID = -1
        self.txtAna = TxtAna()
        self.last_language = "cn"
        # self.total_count = 0

    def recognize_wav(self, mp4_file, language, role, room_id):
        res = {"speak_time": 0,
               "speak_times": 0,
               "speak_speed": 0,
               "tea_praise_times": 0,
               "speak_word_num": 0,
               "asr_result": [],
               "asr_origin": [],
               "talk_with_parent": "",
               "talk_about_homework": "",
               "talk_about_salary": "",
               "talk_with_parent_detail": [],
               "talk_about_homework_detail": [],
               "talk_about_salary_detail": []
               }
        if language != self.last_language:
            self.last_language=language
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 4  # 设置线程数
            opts.inter_op_num_threads = 4  # 设置parallel线程数
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session_encoder = None
            self.onnx_session_decoder = None
            gc.collect()
            if language == "cn":
                self.onnx_session_encoder = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet', 'encoder.onnx'), opts, providers=provider)
                self.onnx_session_decoder = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet', 'decoder.onnx'), opts, providers=provider)
            else:
                self.onnx_session_encoder = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet_en', 'encoder.onnx'),
                                                                         opts, providers=provider)
                self.onnx_session_decoder = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights/wenet_en', 'decoder.onnx'),
                                                                         opts, providers=provider)
        try:
            audio = AudioSegment.from_file(mp4_file, "mp4")  # 课堂音频
            if language == "cn":
                self.mode = "attention_rescoring"
                onnx_session_encoder = self.onnx_session_encoder
                onnx_session_decoder = self.onnx_session_decoder
                vocabulary = self.vocabulary
                session_decoder_input_name_0 = onnx_session_decoder.get_inputs()[0].name
                session_decoder_input_name_1 = onnx_session_decoder.get_inputs()[1].name
                session_decoder_input_name_2 = onnx_session_decoder.get_inputs()[2].name
                session_decoder_input_name_3 = onnx_session_decoder.get_inputs()[3].name
                session_decoder_input_name_4 = onnx_session_decoder.get_inputs()[-1].name
            else:
                onnx_session_encoder = self.onnx_session_encoder
                onnx_session_decoder = self.onnx_session_decoder
                self.mode = "ctc_greedy_search"
                vocabulary = self.vocabulary_en
            session_input_name_1 = onnx_session_encoder.get_inputs()[0].name
            session_input_name_2 = onnx_session_encoder.get_inputs()[1].name
            result = []
            audio = audio.set_frame_rate(16000)  # 没有会出问题
            audio = audio.set_channels(1)  # 没有会出问题
            audio = audio.set_sample_width(2)
            total_time = audio.duration_seconds  # 语音总时长

            chunks = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-45, seek_step=20)  # 结果数组起止点单位为300ms
            chunks_origin = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-75, seek_step=20)  # 结果数组起止点单位为300ms

            if role == 1:  # 判断学生开口是否过长
                speak_time = 0
                for item in chunks:
                    speak_time += (item[1]-item[0])
                speak_time = speak_time//1000
                if speak_time/total_time > 0.3:
                    chunks = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-25, seek_step=20)  # 结果数组起止点单位为300ms
            waveform_list = []
            for item in chunks:
                try:
                    if item[1]-item[0] > 120000:  # 超出2分钟过滤掉
                        continue
                    waveform = audio[item[0]:item[1]]  # ms
                    waveform = np.array(array.array(waveform.array_type, waveform._data))
                    waveform = logfbank(waveform, nfilt=80, lowfreq=20, dither=0.1, wintype='povey')  # kaldi-fbank
                    waveform = np.expand_dims(waveform, axis=0).astype(np.float32)
                    if waveform.shape[1] <= 6:  # 过滤掉时间短的数据
                        continue
                    waveform_list.append((item[0], item[1], waveform))
                except:
                    continue

            for t1, t2, waveform in waveform_list:
                try:
                    speech_lens = np.array([waveform.shape[1]]).astype(np.int32)
                    input = {session_input_name_1: waveform, session_input_name_2: speech_lens}
                    encoder_out, encoder_out_lens, ctc_log_probs, beam_log_probs, beam_log_probs_idx = onnx_session_encoder.run(None, input)  # encoder
                    beam_size = beam_log_probs.shape[-1]
                    batch_size = beam_log_probs.shape[0]
                    if self.mode == 'ctc_greedy_search':  # 英文
                        if beam_size != 1:
                            log_probs_idx = beam_log_probs_idx[:, :, 0]
                        batch_sents = []
                        for idx, seq in enumerate(log_probs_idx):
                            batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
                        hyps = map_batch(batch_sents, vocabulary, self.num_processes, True, 0)
                        if hyps[0].strip() == "":  # 去除空内容
                            continue
                        result.append([t1, t2, hyps[0].strip()])
                        # if index == 0:
                        #     print(hyps[0].strip())  # 英文
                    elif self.mode in ('ctc_prefix_beam_search', "attention_rescoring"):  # 中文
                        batch_log_probs_seq_list = beam_log_probs.tolist()
                        batch_log_probs_idx_list = beam_log_probs_idx.tolist()
                        batch_len_list = encoder_out_lens.tolist()
                        batch_log_probs_seq = []
                        batch_log_probs_ids = []
                        batch_start = []  # only effective in streaming deployment
                        batch_root = TrieVector()
                        root_dict = {}
                        for i in range(len(batch_len_list)):
                            num_sent = batch_len_list[i]
                            batch_log_probs_seq.append(batch_log_probs_seq_list[i][0:num_sent])
                            batch_log_probs_ids.append(batch_log_probs_idx_list[i][0:num_sent])
                            root_dict[i] = PathTrie()
                            batch_root.append(root_dict[i])
                            batch_start.append(True)
                        score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                                   batch_log_probs_ids,
                                                                   batch_root,
                                                                   batch_start,
                                                                   beam_size,
                                                                   self.num_processes,
                                                                   0, -2, 0.99999)
                        if self.mode == 'ctc_prefix_beam_search':
                            hyps = []
                            for cand_hyps in score_hyps:
                                hyps.append(cand_hyps[0][1])
                            hyps = map_batch(hyps, vocabulary, self.num_processes, False, 0)
                    if self.mode == 'attention_rescoring':  # 中文识别
                        ctc_score, all_hyps = [], []
                        max_len = 0
                        for hyps in score_hyps:
                            cur_len = len(hyps)
                            if len(hyps) < beam_size:
                                hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                            cur_ctc_score = []
                            for hyp in hyps:
                                cur_ctc_score.append(hyp[0])
                                all_hyps.append(list(hyp[1]))
                                if len(hyp[1]) > max_len:
                                    max_len = len(hyp[1])
                            ctc_score.append(cur_ctc_score)
                        if self.fp16:
                            ctc_score = np.array(ctc_score, dtype=np.float16)
                        else:
                            ctc_score = np.array(ctc_score, dtype=np.float32)
                        hyps_pad_sos_eos = np.ones(
                            (batch_size, beam_size, max_len + 2), dtype=np.int64) * self.IGNORE_ID
                        r_hyps_pad_sos_eos = np.ones(
                            (batch_size, beam_size, max_len + 2), dtype=np.int64) * self.IGNORE_ID
                        hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
                        k = 0
                        for i in range(batch_size):
                            for j in range(beam_size):
                                cand = all_hyps[k]
                                l = len(cand) + 2
                                hyps_pad_sos_eos[i][j][0:l] = [self.sos] + cand + [self.eos]
                                r_hyps_pad_sos_eos[i][j][0:l] = [self.sos] + cand[::-1] + [self.eos]
                                hyps_lens_sos[i][j] = len(cand) + 1
                                k += 1
                        decoder_ort_inputs = {
                            session_decoder_input_name_0: encoder_out,
                            session_decoder_input_name_1: encoder_out_lens,
                            session_decoder_input_name_2: hyps_pad_sos_eos,
                            session_decoder_input_name_3: hyps_lens_sos,
                            session_decoder_input_name_4: ctc_score}
                        best_index = onnx_session_decoder.run(None, decoder_ort_inputs)[0]
                        best_sents = []
                        k = 0
                        for idx in best_index:
                            cur_best_sent = all_hyps[k: k + beam_size][idx]
                            best_sents.append(cur_best_sent)
                            k += beam_size
                        hyps = map_batch(best_sents, vocabulary, self.num_processes)
                        if hyps[0].strip() != "":  # 去除空内容
                            result.append([t1, t2, hyps[0].strip()])
                            # print("{0}:{1} ".format(int(t1/1000)//60, int(t2/1000) % 60), hyps[0])
                            # print(hyps[0])
                except:
                    continue

            word_num = 0  # 字数
            speak_time = 0  # 开口时长
            speak_times = len(chunks)  # 开口次数
            tea_praise_times = 0  # 夸赞次数
            valid_time = 0  # 有效开口时长，去除少于2个字的句子
            valid_num = 0  # 有效词个数，去除少于1个字的句子
            valid_word_count = 0  # 讲1个字以上的算入语速
            talk_with_parent_detail = ""
            talk_about_homework_detail = ""
            talk_about_salary_detail = ""
            talk_with_parent = '否'
            talk_about_homework = '否'
            talk_about_salary = '否'
            talk_about_salary_index_list=[] #词列表
            for item in result:
                if language == "cn":
                    word_num += len(item[2])
                    if len(item[2]) > 0:
                        speak_time += (item[1]-item[0])
                    current_speed = 1000*len(item[2])/(item[1]-item[0])  # 单句纯语速
                    if len(item[2]) > valid_word_count and current_speed < 10 and current_speed > 1:
                        valid_num += len(item[2])
                        valid_time += (item[1]-item[0]+800)
                else:
                    current_word_num = item[2].count('▁')
                    if current_word_num > 0:
                        speak_time += (item[1]-item[0])
                    word_num += current_word_num
                    current_speed = 1000*current_word_num/(item[1]-item[0])  # 单句纯语速
                    if current_word_num > valid_word_count and current_speed < 10 and current_speed > 1:
                        valid_num += current_word_num
                        valid_time += (item[1]-item[0]+800)
                if role == 0:  # 识别夸赞次数
                    is_praise = self.txtAna.is_praise(item[2])
                    if is_praise:
                        tea_praise_times += 1

                    # 识别讲作业
                    r = self.txtAna.talk_about_homework(item[2])
                    if r:
                        talk_about_homework = '是'
                        talk_about_homework_detail += ('{0}:{1},'.format(math.floor(item[0]/60000), int(item[0]/1000) % 60))
                    # 识别跟家长沟通
                    r = self.txtAna.talk_with_parent(item[2])
                    if r:
                        talk_with_parent = '是'
                        talk_with_parent_detail += ('{0}:{1},'.format(math.floor(item[0]/60000), int(item[0]/1000) % 60))
                    # 识别讨论薪酬
                    r,salary_index = self.txtAna.talk_about_salary(item[2])
                    if r:
                        if salary_index not in talk_about_salary_index_list:
                            talk_about_salary_index_list.append(salary_index)
                        # talk_about_salary = '是'
                        talk_about_salary_detail += ('{0}:{1},'.format(math.floor(item[0]/60000), int(item[0]/1000) % 60))
            if len(talk_about_salary_index_list)>=2:
                talk_about_salary = '是'
            speak_speed = 0 if valid_time == 0 else round(1000*valid_num/valid_time, 2)  # 语速
            res["speak_time"] = speak_time//1000
            res["speak_times"] = speak_times
            res["speak_speed"] = speak_speed
            res["tea_praise_times"] = tea_praise_times
            res["speak_word_num"] = word_num
            res["asr_result"] = result
            res["asr_origin"] = chunks_origin
            res["talk_with_parent"] = talk_with_parent
            res["talk_about_homework"] = talk_about_homework
            res["talk_about_salary"] = talk_about_salary
            res["talk_about_homework_detail"] = talk_about_homework_detail.strip(',')
            res["talk_with_parent_detail"] = talk_with_parent_detail.strip(',')
            res["talk_about_salary_detail"] = talk_about_salary_detail.strip(',')
            return res
        except Exception as ex:
            print("{0}音频分析出错，可能缺少音频文件！{1}".format(room_id, str(ex)))
            return res
