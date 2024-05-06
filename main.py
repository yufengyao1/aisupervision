import os
import gc
import cv2
import time
import math
import json
import copy
import socket
import pynvml
import imageio
import platform
import requests
import warnings
import traceback
import numpy as np
from PIL import Image
import multiprocessing
from YawnAna import YawnDec
from datetime import datetime
from wav_utils_wenet import ASR
from hand_recognize import HandDec
from OneImageAna import OneImageAna
from MicrophoneAna import MicrophoneDec
from pynvml import nvmlDeviceGetMemoryInfo
from kafka import KafkaConsumer, KafkaProducer, TopicPartition, OffsetAndMetadata
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip, CompositeVideoClip
warnings.filterwarnings('ignore')

class VideoAna:
    def __init__(self,is_test):
        self.is_test = is_test
        self.detector = OneImageAna()  # 视频检测引擎
        self.asr = ASR()  # 语音识别引擎
        self.hand_detector = HandDec()
        self.microphone_detector = MicrophoneDec()  # 麦克风检测
        self.yawn_detector = YawnDec()  # 哈欠检测
        self.tea_id = 0  # 老师id
        
        self.tea_img_id = -1  # 老师精彩图片id
        self.stu_img_id = -1  # 学生精彩图片id
        self.stu_img_banner_id = -1  # 学生精彩banner图片id
        self.highlight_id = -1  # 精彩瞬间视频id
        self.class_type = ""  # 课程类型（启蒙）
        self.banner_img = Image.open("src/banner.png")
        self.upload_err_list = []  # 上传失败列表，用于重新上传
        


    def run(self, class_data, worker_id, worker_count=2):  # 分析视频、上传分析结果
        try:
            results = []
            t1 = time.time()
            room_id = class_data['roomId']
            room_type = class_data['roomType']  # 教室类型,0: 1v1、1: 小班课、2: 公开课、3: 教室培训课
            language = 'cn' if (class_data['courseType'] == 0 or class_data['courseType'] == -1) else 'en'
            is_unitcourse = class_data['unitCourse']
            class_time = class_data['replayInfoList'][0]['endTime']//1000-class_data['replayInfoList'][0]['startTime']//1000
            print('--roomid:{0},开始分析_{1},workerid:{2},classtime:{3}s,time:{4}..'.format(room_id, language, worker_id, class_time, time.strftime("%m-%d %H:%M", time.localtime())))
            
            replayInfoList = class_data['replayInfoList']
            replayInfoList = sorted(replayInfoList, key=lambda x: x["role"])  # 排序先分析老师
            global_start_time = replayInfoList[1]["startTime"] if len(replayInfoList) > 1 else replayInfoList[0]["startTime"]  # 课堂正式开始时间戳ms
            global_earliest_time = replayInfoList[0]["startTime"]  # 老师或学生最早进教室的时间-视频回放时间对齐
            for classinfo in replayInfoList[1:]:  # 遍历学生
                if int(classinfo["startTime"]) < global_start_time:  # 查找最早进教室的学生
                    global_start_time = int(classinfo["startTime"])
                if int(classinfo["startTime"]) < global_earliest_time:  # 查找最早进教室的学生或老师
                    global_earliest_time = int(classinfo["startTime"])
            tea_start_time = replayInfoList[0]["startTime"]
            global_start_time = global_start_time if tea_start_time < global_start_time else tea_start_time  # ms
            self.get_classtype(room_id)  # 获取课程类型

            for classinfo in replayInfoList:  # 逐个视频分析
                t_start = time.time()
                mp4_file = ""  # 本地mp4地址
                id = classinfo["userId"]
                if self.is_test:  # 测试模式
                    mp4_file = classinfo["replayUrl"]
                    if classinfo["role"] == 0:
                        self.tea_mp4_file = classinfo["replayUrl"]
                        self.tea_id = id
                else:  # 生产模式
                    # if classinfo["role"] != 0 and language=="en": #跳过英文学生
                    #     continue
                    
                    if classinfo["role"] == 0:
                        self.tea_mp4_file = mp4_file
                        self.tea_id = id
                    if not r:  # 视频下载失败
                        return results
                result = self.analyse_video(mp4_file, room_id, room_type, classinfo, global_start_time, global_earliest_time,
                                            worker_id=worker_id, language=language, unitcourse=int(is_unitcourse))  # 分析单个视频，cn-中文,en-英文

                # 计算成本
                current_class_time = (classinfo['endTime']-classinfo['startTime'])//1000  # 当前课程时长s
                current_class_time = 3300 if current_class_time == 0 else current_class_time
                t_end = time.time()
                t_interval = t_end-t_start  # 分析用时
                price = 460*6.5*t_interval/current_class_time/720/worker_count  # 计算单价
                result["price"] = round(price, 4)
                print('{0}-price:{1}'.format(worker_id, result["price"]))
                results.append(result)
            results = self.cal_stu_score(results)  # 所有视频分析完后，计算学生得分
            results = self.cal_tea_score(results)  # 所有视频分析完后，计算老师得分

            t2 = time.time()
            print("############worker:{0},roomid:{1},分析完成,用时{2}秒...".format(worker_id, room_id, int(t2-t1)))
            return results
        except Exception as ex:
            traceback.print_exc()
            return []
        finally:
            for classinfo in replayInfoList:  # 删除视频
                id = classinfo["userId"]
                mp4_file = os.path.join(os.path.dirname(__file__), 'tmp/{0}_{1}.mp4'.format(room_id, id))
                if os.path.exists(mp4_file) and not self.is_test:  # 删除本地视频
                    os.remove(mp4_file)

    def analyse_video(self, mp4_file, room_id, room_type, classinfo, global_start_time=0, global_earliest_time=0, worker_id='1', language="cn", unitcourse=0):  # jump_time 表示跳过视频开始n秒
        happy_img_file = os.path.join(os.path.dirname(__file__), 'tmp/highlights/happy_'+worker_id+'.jpg')  # 精彩图片路径
        happy_img_banner_file = os.path.join(os.path.dirname(__file__), 'tmp/highlights/happy_banner_'+worker_id+'.jpg')  # 精彩图片路径
        highlight_file = os.path.join(os.path.dirname(__file__), 'tmp/highlights/highlight_'+worker_id+'.mp4')  # 精彩视频路径-学生
        history_frames = []  # 历史帧
        max_emo_score = 0  # 最佳表情得分
        max_emo_frame_rgb = None  # 最佳表情frame
        last_leave_index = -1  # 上一次离屏时间s
        is_leaving = False  # 离屏中
        last_otherpeopele_index = -1  # 上一次他人干扰时间s
        last_eyeclose_index = -1  # 上次闭眼index
        is_othering = False  # 正在干扰中
        role = classinfo["role"]  # 角色
        num_analysed = 0  # 分析的帧数
        brightness_list = []  # 亮度
        microphone_count = 0  # 戴麦克风次数
        microphone_ana_times = 0
        current_eye_close_num = 0
        is_yawnning = False  # 开始哈欠
        mouse_list = []
        yawnData = []  # 开口数据
        yawn_frames = []  # 哈欠帧
        face_count_list = []
        last_yawn_index = -1
        last_yawn_end_index = -100  # 上一次哈欠结束时间，两次哈欠间隔5秒以上
        start_space_time = (int(classinfo['startTime'])-global_earliest_time)/1000  # 开头黑屏时间(进教室晚)
        jump_time = global_start_time-int(classinfo['startTime'])  # 跳过视频开头时长ms
        jump_time = 0 if jump_time < 0 else jump_time
        self.tea_start_time = int(classinfo['startTime'])/1000 if role == 0 else self.tea_start_time  # 老师课程开始时间戳s-用于视频对齐
        leave_time_span = 5 if role == 0 else 10  # 离开时间间隔阈值
        is_doing_ges = [False]*19
        last_gesture_index = 0

        result = {
            "roomId": room_id,  # 房间号
            "roomType": room_type,  # 教室类型,0: 1v1、1: 小班课、2: 公开课、3: 教室培训课
            "language": language,
            "role": role,  # 0-teacher
            "userId": classinfo["userId"],  # id
            "userName": classinfo["userName"] if 'userName' in list(classinfo.keys()) else "",
            "startTime": classinfo['startTime'],  # 视频开始时间
            "endTime": classinfo['endTime'],  # 视频结束时间
            "close_eye_time_total": 0,  # 所有闭眼累计时长
            "extra": {
                "brightness_val": 150,  # 房间亮度值
                "brightness": "unknown",  # 房间亮度
                "have_microphone": 1,  # 戴麦克风 0-没戴，1-戴
                "close_eye_time": 0,  # 长时间闭眼总时长
                "close_eye_list": "",  # 长时间闭眼时间节点
                "close_eye_times": 0,  # 长时间闭眼次数
                "close_eye_percent": 0,  # 总闭眼时长占比
                "yawn_times": 0,  # 哈欠次数
                "yawn_details": "",  # 哈欠详情
                "gesture_details": "",  # 手势激励详情
                "pos_score": 0,  # 姿态得分
                "face_percent": 0,  # 露脸占比
                "tea_score_2": 0,  # 老师分数-受学生影响
                "gesture_time": 0,
                "gesture_times": 0,
                "h_emo_score": 0,  # 精彩瞬间表情得分
                "h_speak_score": 0,  # 精彩瞬间开口得分
                "unitcourse": '是' if unitcourse == 1 else '否',  # 是否单元课 0-否 1-是
                "talk_about_homework": '否',  # 是否讲到作业课
                "talk_about_homework_detail": [],  # 作业课关键词列表
                "talk_with_parent": '否',  # 是否跟家长交流
                "talk_with_parent_detail": [],  # 跟家长交流列表
                "talk_about_salary": '否',  # 是否讲到薪酬
                "talk_about_salary_detail": [],  # 薪酬关键词列表
                "tea_id": self.tea_id,  # 老师id
                "asr_list": [],
                "happy_score":0, #高兴得分
                "happy_rate":0, #高兴占比 
            },
            "face_count": 0,  # 检测到的人脸个数
            "pos_scores": [],  # 姿态得分
            "class_time": (classinfo['endTime']-global_start_time)//1000,  # 课程时长
            "leave_time": 0,  # 离屏时长
            "leave_percent": 0,  # 离屏时长百分比
            "leave_time_list": "",  # 老师离开时间节点
            "other_people": 0,  # 他人干扰时长s
            "other_people_list": "",  # 他人干扰时间节点
            "speak_time": 500,  # 开口时长500s
            "speak_times": 10,  # 开口次数
            "speak_time_percent": 0,  # 开口时长占比
            "speak_speed": 3.2,  # 语速 3.2字/秒
            "speak_word_num": 0,  # 讲话字数
            "emo_score": 0,  # 综合表情得分ccc
            "emotion": [],  # 每秒的表情得分数组
            "gesture_encourage_times": 0,  # 手势激励次数
            "gesture_encourage_time": 0,  # 手势激励时长
            "img_url": "",  # 精彩瞬间照片
            "highlights": "",  # 学生精彩视频
            "highlights_tea": "",  # 老师精彩视频
            "highlight_score": 0,  # 学生精彩瞬间得分
            "total_score": 0,  # 得分
            "playback_url": "https://classroom.lingoace.com/replay/"+str(room_id)+"/"+str(classinfo['recordId'])  # 视频回放地址
        }
        try:
            # ---------------------------------------------------------------------------------------语音分析
            asr_res=self.asr.recognize_wav(mp4_file, language, role, room_id)
            result["speak_time"]=asr_res["speak_time"]
            result["speak_times"]=asr_res["speak_times"]
            result["speak_speed"]=asr_res["speak_speed"]
            result["speak_word_num"]=asr_res["speak_word_num"]
            result["tea_praise_times"]=asr_res["tea_praise_times"]
            result["extra"]["talk_with_parent"]=asr_res["talk_with_parent"]
            result["extra"]["talk_about_homework"]=asr_res["talk_about_homework"]
            result["extra"]["talk_about_salary"]=asr_res["talk_about_salary"]
            result["extra"]["talk_about_homework_detail"]=asr_res["talk_about_homework_detail"]
            result["extra"]["talk_with_parent_detail"]=asr_res["talk_with_parent_detail"]
            result["extra"]["talk_about_salary_detail"]=asr_res["talk_about_salary_detail"]
            asr_result=asr_res["asr_result"]
            asr_origin=asr_res["asr_origin"]
    
            asr_dict = {} #讲话内容
            if language == "cn":
                for t_s, t_e, txt in asr_result:
                    if len(txt) > 2:
                        asr_dict[str(t_s)] = txt
            else:
                for t_s, t_e, txt in asr_result:
                    if len(txt) > 2:
                        asr_dict[str(t_s)] = txt
            result["extra"]["asr_list"] = asr_dict  # 存储讲话结果

            # ---------------------------------------------------------------------------------------视频分析
            reader = imageio.get_reader(mp4_file, 'ffmpeg')  # 课堂视频
            real_fps = reader.get_meta_data()['fps']  # 视频帧率
            jump_num = jump_time*real_fps/1000  # 需要跳过的视频开头帧数
            history_frame_num = 3*real_fps  # 历史缓存帧数

            for index, frame_rgb in enumerate(reader):  # 帧遍历
                if len(history_frames) < history_frame_num:
                    history_frames.append(frame_rgb)
                else:
                    history_frames.pop(0)
                    history_frames.append(frame_rgb)

                analyse_this = (index == math.ceil(jump_num+real_fps*num_analysed))  # 是否分析本帧

                if (not analyse_this) and (not is_yawnning):  # 非待检测帧和开头静默段跳过
                    continue
                elif analyse_this:
                    num_analysed += 1  # 检测帧数递增

                max_emo_frame_rgb = frame_rgb if max_emo_frame_rgb is None else max_emo_frame_rgb  # 默认设置首张图片为精彩图片
                res, face_count = self.detector.getResult(frame_rgb, detect_eye=True, detect_emo=True, detect_keypoints=True, detect_age=False)  # 分析单张图像

                # -----------------------------------------------------------哈欠分析(非逐秒分析)
                if res:  # 非逐秒哈欠分析
                    mouse = res["mouse"]
                    mouse_dis = res["mouse_dis"]
                    if is_yawnning:  # 持续哈欠
                        yawn_frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        yawnData.append(mouse)
                        if mouse < 0.3:  # 哈欠结束
                            is_yawnning = False
                            is_yawn = self.get_yawn_result(yawnData, yawn_frames)  # 哈欠分析
                            if is_yawn:
                                max_mouse_index = np.argmax(yawnData)
                                is_speaking = self.is_speaking(1000*(last_yawn_index+max_mouse_index)/real_fps, asr_result)  # 哈欠最大处是否讲话
                                if not is_speaking:
                                    result["extra"]["yawn_times"] += 1
                                    result["extra"]["yawn_details"] += '{0}:{1},'.format(math.floor((last_yawn_index/real_fps+start_space_time)/60),
                                                                                         int(last_yawn_index/real_fps+start_space_time) % 60)
                                    last_yawn_end_index = index
                            yawnData.clear()
                            yawn_frames.clear()
                    elif mouse > 0.45 and mouse_dis > 10 and index-last_yawn_end_index > 5*real_fps:  # 首次哈欠
                        is_yawnning = True
                        for h_i in range(len(history_frames)):
                            frame_rgb_tmp = history_frames[len(history_frames)-1-h_i]
                            res_tmp, _ = self.detector.getResult(frame_rgb_tmp, detect_eye=False, detect_emo=False, detect_keypoints=True)  # 分析单张图像
                            yawn_frames.append(cv2.cvtColor(frame_rgb_tmp, cv2.COLOR_RGB2BGR))
                            if res_tmp is None:
                                yawnData.append(0)
                                break
                            elif res_tmp["mouse"] <= 0.25:
                                yawnData.append(res_tmp["mouse"])
                                break
                            else:
                                yawnData.append(res_tmp["mouse"])
                        yawnData.reverse()
                        yawn_frames.reverse()
                        last_yawn_index = index-len(yawn_frames)+1
                else:  # 结束哈欠
                    is_yawnning = False
                    yawnData.append(0)
                    yawn_frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    is_yawn = self.get_yawn_result(yawnData, yawn_frames)  # 哈欠分析
                    if is_yawn:
                        max_mouse_index = np.argmax(yawnData)
                        is_speaking = self.is_speaking(1000*(last_yawn_index+max_mouse_index)/real_fps, asr_result)
                        if not is_speaking:
                            result["extra"]["yawn_times"] += 1
                            result["extra"]["yawn_details"] += '{0}:{1},'.format(math.floor((last_yawn_index/real_fps+start_space_time)/60), int(last_yawn_index/real_fps+start_space_time) % 60)
                            last_yawn_end_index = index
                    yawnData.clear()
                    yawn_frames.clear()
                if (not analyse_this):  # 只逐帧分析打哈欠
                    continue

                # -----------------------------------------------------------逐秒分析
                result["face_count"] = result["face_count"]+1 if face_count != 0 else result["face_count"]  # face_count是必返回参数
                face_count_list.append(face_count)
                if res:  # 结果分析
                    # age_list.append(res["age"])  # 年龄列表
                    pos_score = res["pos_score"]  # 坐姿得分
                    result["pos_scores"].append(pos_score)
                    mouse_list.append(res["mouse"])
                    # -----------------------------------------------------------闭眼分析
                    if not res["open_eye"]:
                        current_eye_close_num += 1  # 当前闭眼次数
                        result["close_eye_time_total"] += 1  # 总闭眼时长
                        if last_eyeclose_index == -1:
                            last_eyeclose_index = index
                    else:
                        if current_eye_close_num >= 3:
                            if self.is_silence(1000*last_eyeclose_index/real_fps, 1000*index/real_fps, asr_origin):  # 判断老师是否讲话
                                eye_opened = False
                                open_num = 0
                                for frame_rgb_tmp in history_frames[1:-1]:  # 分析3秒内是否有睁眼
                                    res_tmp, _ = self.detector.getResult(frame_rgb_tmp, detect_eye=True, detect_emo=False, detect_keypoints=False)  # 分析单张图像
                                    if res_tmp is None:
                                        continue
                                    elif res_tmp["open_eye"]:
                                        open_num += 1
                                        if open_num > 3*real_fps/5:
                                            eye_opened = True
                                            break

                                if not eye_opened:
                                    result["extra"]["close_eye_times"] += 1  # 闭眼超过5秒钟次数
                                    result["extra"]["close_eye_time"] += current_eye_close_num  # 总闭眼时长
                                    result["extra"]["close_eye_list"] += '{0}:{1},'.format(math.floor((last_eyeclose_index/real_fps+start_space_time)/60),
                                                                                           int(last_eyeclose_index/real_fps+start_space_time) % 60)

                        current_eye_close_num = 0
                        last_eyeclose_index = -1
                    # -----------------------------------------------------------表情分析
                    emo_des, emo_s = res["emo"]
                    if emo_s is not None:
                        # emo_score = 100*(0.059785*emo[1][0]+0.027026*emo[1][1]+0.039953*emo[1][2]+0.368842*emo[1][3]+0.059785 * emo[1][4]+0.260724*emo[1][5]+0.183885*emo[1][6])/0.368842  # 表情得分,不可舍入
                        if role == 0:
                            emo_score_origin = 16.2088*emo_s[0]+7.3273*emo_s[1]+10.832*emo_s[2]+110*emo_s[3]+16.2088 * emo_s[4]+70.6872 * \
                                emo_s[5]+49.85468*emo_s[6]  # 原始打分[0-'Angry', 1-'Disgust', 2-'Fear', 3-'Happy', 4-'Sad', 5-'Surprise', 6-'Neutral']
                        else:
                            emo_score_origin = 16.2088*emo_s[0]+7.3273*emo_s[1]+10.832*emo_s[2]+110*emo_s[3]+16.2088 * emo_s[4]+50 * \
                                emo_s[5]+49.85468*emo_s[6]
                        emo_score = 100 if emo_score_origin > 100 else emo_score_origin
                        result["emotion"].append(int(emo_score))
                        result["extra"]["happy_score"]+=100*emo_s[3]
                        if emo_s[3]>0.5:
                            result["extra"]["happy_rate"]+=1
                        yaw, pitch, roll = res["face_angle"]  # 计算头部姿态得分
                        head_pos_score = 100-2*abs(yaw)
                        head_pos_score=0 if head_pos_score<0 else head_pos_score
                        emo_pos_score = 0.85*emo_score_origin+0.05*pos_score+0.1*head_pos_score  # 表情和姿态综合得分
                        if emo_pos_score > max_emo_score and face_count == 1:  # 表情最佳画像-限制单个人脸
                            max_emo_score = emo_pos_score
                            max_emo_frame_rgb = frame_rgb
                            # print('当前最佳表情得分:{0},坐姿得分:{1},总分:{2}'.format(emo_score_origin,pos_score,emo_pos_score))
                    else:
                        result["emotion"].append(0)
                else:  # 不存在表情
                    mouse_list.append(0)
                    current_eye_close_num = 0  # 闭眼时长归零
                    result["emotion"].append(0)
                    result["pos_scores"].append(np.nan)
                # ---------------------------------------------------------------离屏分析
                if face_count == 0:  # 看不到人
                    is_speaking = self.is_speaking(1000*index/real_fps, asr_origin)
                    if not is_speaking:  # 没在讲话
                        last_leave_index = index if not is_leaving else last_leave_index  # 上次离屏时间
                        is_leaving = True
                    else:  # 在讲话
                        if is_leaving and index-last_leave_index >= leave_time_span*real_fps:  # 出现人脸
                            result["leave_time_list"] += '{0}:{1},'.format(math.floor((last_leave_index/real_fps+start_space_time)/60), int(last_leave_index/real_fps+start_space_time) % 60)
                            result["leave_time"] += int((index-last_leave_index)//real_fps)
                            is_leaving = False
                elif face_count >= 1:
                    if is_leaving and index-last_leave_index >= leave_time_span*real_fps:  # 出现人脸
                        # if self.is_silence(1000*last_leave_index/real_fps, 1000*index/real_fps, asr_origin, time_span=3000):
                        result["leave_time_list"] += '{0}:{1},'.format(math.floor((last_leave_index/real_fps+start_space_time)/60), int(last_leave_index/real_fps+start_space_time) % 60)
                        result["leave_time"] += int((index-last_leave_index)//real_fps)
                    is_leaving = False
                # ---------------------------------------------------------------他人干扰分析
                if face_count > 1:  # 他人干扰
                    last_otherpeopele_index = index if not is_othering else last_otherpeopele_index
                    is_othering = True  # 他人干扰中
                else:  # 无他人干扰
                    if is_othering and index-last_otherpeopele_index >= 5*real_fps:
                        result["other_people_list"] += '{0}:{1},'.format(math.floor((last_otherpeopele_index/real_fps+start_space_time)/60),
                                                                         int(last_otherpeopele_index/real_fps+start_space_time) % 60)
                        result["other_people"] += int((index-last_otherpeopele_index)//real_fps)
                    is_othering = False

                if role == 0:  # 手势激励-老师，单帧0.015s
                    # ---------------------------------------------------------------光线亮度分析
                    if num_analysed % 10 == 0:
                        brightness_list.append(self.cal_lightness(frame_rgb))
                    # ---------------------------------------------------------------手势分析
                    gesture_encourage_list = [6, 16, 17]
                    gesture_name, ges_index = self.hand_detector.detect(frame_rgb)
                    if ges_index != 0:  # 出现教学手势
                        result['extra']["gesture_time"] += 1  # 教学手势时长加1
                        if not is_doing_ges[ges_index]:  # 新出现此类手势
                            last_gesture_time = index
                            is_doing_ges[ges_index] = True  # 正在比划新手势
                            if last_gesture_index != 0:  # 上一次手势也是教学手势
                                if last_gesture_index in gesture_encourage_list:
                                    result["gesture_encourage_times"] += 1  # 手势激励次数
                                result['extra']["gesture_times"] += 1  # 教学手势次数加1
                                result["extra"]["gesture_details"] += '{0}:{1},'.format(math.floor((last_gesture_time / real_fps+start_space_time)/60),
                                                                                        int(last_gesture_time/real_fps+start_space_time) % 60)
                                is_doing_ges[last_gesture_index] = False
                        last_gesture_index = ges_index
                    else:
                        if last_gesture_index != 0:
                            if last_gesture_index in gesture_encourage_list:
                                result["gesture_encourage_times"] += 1  # 手势激励次数
                            result['extra']["gesture_times"] += 1  # 教学手势次数加1
                            result["extra"]["gesture_details"] += '{0}:{1},'.format(math.floor((last_gesture_time / real_fps+start_space_time)/60),
                                                                                    int(last_gesture_time/real_fps+start_space_time) % 60)
                            is_doing_ges[last_gesture_index] = False
                        last_gesture_index = 0

                    # ---------------------------------------------------------------麦克风分析
                    if num_analysed % 10 == 0 and face_count > 0:
                        index_pred, softmax_value = self.microphone_detector.detect(frame_rgb)  # 麦克风检测
                        microphone_ana_times += 1
                        if index_pred == 0:
                            microphone_count += 1

            if is_leaving and index-last_leave_index >= leave_time_span*real_fps:  # 视频结束，判断最后一段时间的离开情况
                result["leave_time_list"] += '{0}:{1},'.format(math.floor((last_leave_index/real_fps+start_space_time)/60), int(last_leave_index/real_fps+start_space_time) % 60)
                result["leave_time"] += int((index-last_leave_index)//real_fps)
            if self.is_test:
                result["class_time"] = num_analysed  # 测试环境下没有准确的时间值，需要用num_analysed代替时间

            if max_emo_frame_rgb is None:
                print('error: max_emo_frame is none!')
            else:
                max_emo_frame = cv2.cvtColor(max_emo_frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(happy_img_file, max_emo_frame)  # 保存精彩图片

                # 正式环境上传到标准服务
                if role == 0:
                    
                    self.tea_img_id = img_id
                else:
                    
                    self.stu_img_id = img_id

                    # banner_img = self.banner_image(max_emo_frame, self.banner_img.copy())
                    # banner_img.save(happy_img_banner_file)
                    # self.stu_img_banner_id = img_id

                

            result["speak_time_percent"] = 0 if result["class_time"] == 0 else round(result["speak_time"]/result["class_time"], 4)  # 开口时长占比(class_time测试模式为实际检测帧率，不能放图像分析前面)

            # 精彩瞬间提取-依赖语音识别
            if role == 1:  # 学生
                stu_start_time = int(classinfo['startTime'])/1000  # 学生课程开始时间s
                interval_offset = self.tea_start_time-stu_start_time  # 老师与学生课程时间差s
                highlight_list, highlight_score, h_emo_score, h_speak_score, create_highlight = self.highlight_extract(
                    asr_result, result["emotion"], face_count_list, result["pos_scores"], mouse_list, math.floor(jump_num))  # 精彩时刻列表
                result["extra"]["h_emo_score"], result["extra"]["h_speak_score"] = h_emo_score, h_speak_score  # 精彩瞬间开口及表情得分

                # if True:
                if create_highlight:
                    print("{0}开始合成精彩瞬间视频...".format(room_id))
                    url_highlight, id_highlight = self.get_highlight_url_composed(room_id, classinfo["userId"], highlight_list, mp4_file, highlight_file, interval_offset)
                    if id_highlight is not None and id_highlight != -1 and self.tea_img_id != -1 and self.stu_img_id != -1:
                        data = {"roomId": room_id, "userId": classinfo["userId"], "roleType": 1, "movFileId": id_highlight, "tutorImgFileId": self.tea_img_id,
                                "imgFileId": self.stu_img_id, "bannerImgFileId": self.stu_img_banner_id, "score": math.floor(highlight_score)}
                        if not is_success:  # 上传失败
                            self.upload_err_list.append(copy.deepcopy(data))
                            print("{0}精彩瞬间上传失败！".format(room_id))
                        else:  # 上传成功，把历史失败重新上传一遍
                            print("{0}精彩瞬间上传成功！".format(room_id))
                            err_len = len(self.upload_err_list)  # 指定重新上传的最大个数
                            for i in range(err_len):
                                data = self.upload_err_list.pop(0)
                                if not is_success:
                                    self.upload_err_list.append(copy.deepcopy(data))
                                    print("{0}上传失败！".format(room_id))
                    else:
                        print("{0}不符合上传条件!id_highlight:{1}, tea_img_id:{2}, stu_img_id:{3}, stu_img_banner_id:{4}".format(
                            room_id, id_highlight, self.tea_img_id, self.stu_img_id, self.stu_img_banner_id))

                    result["highlights"] = url_highlight
                    result["highlight_score"] = math.floor(highlight_score)
                else:
                    result["highlights"] = ""
                    result["highlight_score"] = math.floor(highlight_score)

            # 坐姿计算
            if len(result["pos_scores"]) == 0:
                pos_score = 0
            else:
                pos_score = np.nanmean(result["pos_scores"])
                pos_score = 0 if math.isnan(pos_score) else int(pos_score)
            if role == 0:  # 老师教室亮度和戴麦统计
                brightness = np.mean(brightness_list) if len(brightness_list) > 0 else 0
                result["extra"]["brightness_val"] = int(brightness)
                if brightness < 80:
                    result["extra"]["brightness"] = "过暗"
                elif brightness < 150:
                    result["extra"]["brightness"] = "偏暗"
                elif brightness < 220:
                    result["extra"]["brightness"] = "正常"
                else:
                    result["extra"]["brightness"] = "过亮"
                if microphone_ana_times == 0:
                    result["extra"]["have_microphone"] = 1
                else:
                    result["extra"]["have_microphone"] = 1 if microphone_count/microphone_ana_times > 0.5 else 0  # 麦克风

            result["extra"]["pos_score"] = pos_score
            result["leave_percent"] = 0 if result["class_time"] == 0 else round(result["leave_time"]/result["class_time"], 4)  # 离屏占比
            result["extra"]["face_percent"] = 0 if result["class_time"] == 0 else round(result["face_count"]/result["class_time"], 4)  # 露脸占比
            result["extra"]["face_percent"] = 1 if result["extra"]["face_percent"] > 1 else result["extra"]["face_percent"]
            result["extra"]["close_eye_percent"] = 0 if result["class_time"] == 0 else round(result["extra"]["close_eye_time"]/result["class_time"], 4)  # 闭眼占比
            emoscore = np.array(result["emotion"])  # 表情得分
            emoscore = emoscore[emoscore > 0]  # 过滤离屏情况
            if len(emoscore)>0:
                result["extra"]["happy_score"]=round(result["extra"]["happy_score"]/len(emoscore),4)
                result["extra"]["happy_rate"]=round(result["extra"]["happy_rate"]/len(emoscore),4)
                result["emo_score"] = int(np.mean(emoscore))  # 表情得分
            result["leave_time_list"] = result["leave_time_list"].strip(',')
            result["other_people_list"] = result["other_people_list"].strip(',')
            result["extra"]["close_eye_list"] = result["extra"]["close_eye_list"].strip(',')
            result["extra"]["gesture_details"] = result["extra"]["gesture_details"].strip(',')
            result["extra"]["yawn_details"] = result["extra"]["yawn_details"].strip(',')

            return result
        except Exception as ex:
            traceback.print_exc()
            print('视频分析出错，可能缺少视频！{0}'.format(ex))
            return result

    def banner_image(self, back_img: cv2.Mat, fore_img: Image) -> Image:  # (Mat, PIL Image)
        h, w = back_img.shape[0], back_img.shape[1]
        if w == h:
            hh = int(240*w/320)
            hh = int((h-hh)/2)
            back_img = back_img[hh:h-hh, :]
        scale = 564/back_img.shape[1]
        back_img = cv2.resize(back_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        back_img = cv2.copyMakeBorder(back_img, 22, 20, 230, 197, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        back_img = back_img[:460, :990]
        bg = Image.fromarray(cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB))
        bg.paste(fore_img, (0, 0), mask=fore_img)
        return bg

    def get_yawn_result(self, yawnData, yawn_frames):
        isyawn = True
        if len(yawnData) <= 20:  # 1.哈欠过短或过长
            return False
        elif yawnData[0] > 0.3:  # 2.无左侧闭口
            return False
        elif yawnData[0] == 0 and yawnData[-1] == 0:  # 4.两边截断
            return False
        elif np.max(yawnData) < 0.55:  # 3.哈欠阈值
            return False
        if len(yawn_frames) > 90:
            yawn_frames = yawn_frames[:90]
        index_pred, softmax_value = self.yawn_detector.detect(yawn_frames)
        isyawn = True if index_pred == 1 else False
        return isyawn

    def is_silence(self, start_time, end_time, asr_result, time_span=5000):  # 判断时间段附近是否有人讲话
        left = 0
        right = 0
        for index, item in enumerate(asr_result[:-1]):
            space_left = item[1]
            space_right = asr_result[index+1][0]
            if end_time < asr_result[0][0]-5000:
                return True
            elif start_time > asr_result[-1][1]+5000:
                return True
            if start_time <= space_left:
                left = space_left
                if end_time <= space_left:
                    continue
                elif end_time <= space_right:
                    right = end_time
                else:
                    right = space_right
            elif start_time >= space_left and start_time < space_right:
                left = start_time
                if end_time <= space_right:
                    right = end_time
                else:
                    right = space_right
            else:
                continue

            if right-left > time_span:  # 静默消失大于5秒
                return True
        return False

    def is_speaking(self, time, asr_result):
        is_speaking = False
        try:
            for index, item in enumerate(asr_result[:-1]):
                if time >= item[0] and time <= item[1]:
                    is_speaking = True
                    break
        except Exception as ex:
            print(ex)
        return is_speaking

    def cal_tea_score(self, results):  # 老师课堂打分
        stu_speak_score = 0
        tea_emo_score = 0
        tea_praise_score = 0
        tea_speak_speed_score = 0
        stu_speak_percent = 0
        stu_names = ""  # 学生名字列表，中间用逗号隔开
        stu_ids = ""  # 学生id列表，中间用逗号隔开
        tea_praise_level = "A"
        tea_speak_speed_level = "A"
        stu_speak_level = "A"
        stu_count = 0  # 学生个数
        stu_ave_score = 0  # 学生平均分
        for result in results:
            if result["role"] == 1:  # 学生
                stu_names += (result["userName"]+",")
                stu_ids += (str(result["userId"])+",")
                if result["class_time"] != 0:
                    stu_speak_percent += (result["speak_time"]/result["class_time"])
                    stu_ave_score += result["total_score"]
                    stu_count += 1
        stu_names = stu_names.strip(',')
        stu_ids = stu_ids.strip(',')
        stu_ave_score = stu_ave_score/stu_count if stu_count > 0 else -1

        for result in results:
            if result["role"] == 0:  # 老师
                class_time = result["class_time"]  # 课程时长
                tea_emo_score = result["emo_score"]  # 表情得分
                praise_percent = (result["tea_praise_times"]+result["gesture_encourage_times"])/class_time/2 if class_time != 0 else 0  # 表扬次数
                if praise_percent >= 0.01:
                    tea_praise_level = "A"
                    tea_praise_score = 100
                elif praise_percent >= 0.002:
                    tea_praise_level = "B"
                    tea_praise_score = 5000*praise_percent+50
                else:
                    tea_praise_level = "C"
                    tea_praise_score = 30000*praise_percent
                tea_praise_score = 100 if tea_praise_score > 100 else tea_praise_score
                tea_praise_score = 0 if tea_praise_score < 0 else tea_praise_score

                speak_speed = result["speak_speed"]  # 语速
                if '启蒙' in self.class_type:  # 启蒙课语速
                    if speak_speed <= 1.8:
                        tea_speak_speed_score = 75*speak_speed-75
                        tea_speak_speed_level = "C"
                    elif speak_speed > 1.8 and speak_speed < 2.2:
                        tea_speak_speed_score = 100*speak_speed-120  # 60
                        tea_speak_speed_level = "B"
                    elif speak_speed >= 2.2 and speak_speed <= 3.2:
                        tea_speak_speed_score = 100
                        tea_speak_speed_level = "A"
                    elif speak_speed > 3.2 and speak_speed < 3.5:
                        tea_speak_speed_score = -400*speak_speed/3+100+3.2*400/3
                        tea_speak_speed_level = "B"
                    else:
                        tea_speak_speed_score = 480-speak_speed*120
                        tea_speak_speed_level = "C"
                    # tea_speak_speed_score = 100 if tea_speak_speed_score > 100 else tea_speak_speed_score
                    # tea_speak_speed_score = 0 if tea_speak_speed_score < 0 else tea_speak_speed_score
                elif '进阶' in self.class_type:  # 进阶课语速
                    if speak_speed <= 2.2:
                        tea_speak_speed_score = 300*speak_speed-600
                        tea_speak_speed_level = "C"
                    elif speak_speed > 2.2 and speak_speed < 2.6:
                        tea_speak_speed_score = 50*speak_speed-30  # 60
                        tea_speak_speed_level = "B"
                    elif speak_speed >= 2.6 and speak_speed <= 3.6:
                        tea_speak_speed_score = 100
                        tea_speak_speed_level = "A"
                    elif speak_speed > 3.6 and speak_speed < 4.2:
                        tea_speak_speed_score = 220-100*speak_speed/3
                        tea_speak_speed_level = "B"
                    elif speak_speed >= 4.2:
                        tea_speak_speed_score = 375-speak_speed*75
                        tea_speak_speed_level = "C"
                elif result["language"]=="en": #英文课
                    if speak_speed <= 1.8:
                        tea_speak_speed_score = 75*speak_speed-75
                        tea_speak_speed_level = "C"
                    elif speak_speed > 1.8 and speak_speed < 2.2:
                        tea_speak_speed_score = 100*speak_speed-120  # 60
                        tea_speak_speed_level = "B"
                    elif speak_speed >= 2.2 and speak_speed <= 3.2:
                        tea_speak_speed_score = 100
                        tea_speak_speed_level = "A"
                    elif speak_speed > 3.2 and speak_speed < 3.5:
                        tea_speak_speed_score = -400*speak_speed/3+100+3.2*400/3
                        tea_speak_speed_level = "B"
                    else:
                        tea_speak_speed_score = 480-speak_speed*120
                        tea_speak_speed_level = "C"
                else:  # 其它课语速
                    if speak_speed <= 2.2:
                        tea_speak_speed_score = 300*speak_speed-600
                        tea_speak_speed_level = "C"
                    elif speak_speed > 2.2 and speak_speed < 2.5:
                        tea_speak_speed_score = 400*speak_speed/3+100-1000/3  # 60
                        tea_speak_speed_level = "B"
                    elif speak_speed >= 2.5 and speak_speed <= 3.3:
                        tea_speak_speed_score = 100
                        tea_speak_speed_level = "A"
                    elif speak_speed > 3.3 and speak_speed < 3.7:
                        tea_speak_speed_score = 430-100*speak_speed
                        tea_speak_speed_level = "B"
                    else:
                        tea_speak_speed_score = 800-speak_speed*200
                        tea_speak_speed_level = "C"
                tea_speak_speed_score = 100 if tea_speak_speed_score > 100 else tea_speak_speed_score
                tea_speak_speed_score = 0 if tea_speak_speed_score < 0 else tea_speak_speed_score

                if stu_speak_percent == 0:
                    stu_speak_level = "unknown"
                elif stu_speak_percent >= 0.19 and stu_speak_percent <= 0.24:
                    stu_speak_level = "A"
                elif stu_speak_percent <= 0.11 or stu_speak_percent >= 0.38:
                    stu_speak_level = "C"
                else:
                    stu_speak_level = "B"

                if stu_speak_percent >= 0.19 and stu_speak_percent <= 0.24:
                    stu_speak_score = 100
                elif stu_speak_percent < 0.19:
                    stu_speak_score = 526.32*stu_speak_percent
                elif stu_speak_percent > 0.24:
                    stu_speak_score = 201-421*stu_speak_percent
                stu_speak_score = 100 if stu_speak_score > 100 else stu_speak_score
                stu_speak_score = 0 if stu_speak_score < 0 else stu_speak_score

                score = 0.38*tea_emo_score+0.24*tea_praise_score+0.38*tea_speak_speed_score  # 三个指标打分
                if stu_ave_score != -1:
                    score_2 = 0.3*tea_emo_score+0.2*tea_praise_score+0.3*tea_speak_speed_score+0.2*stu_ave_score  # 四个指标打分
                else:
                    score_2 = score

                if result["roomType"] == 1:  # 小班课
                    score_level = "5"
                    if score < 30:
                        score_level = "1"
                    elif score > 30 and score <= 40:
                        score_level = "2"
                    elif score > 40 and score <= 70:
                        score_level = "3"
                    elif score > 70 and score <= 85:
                        score_level = "4"
                    elif score > 85:
                        score_level = "5"
                else:
                    score_level = "5"
                    if score <= 40:
                        score_level = "1"
                    elif score > 40 and score <= 55:
                        score_level = "2"
                    elif score > 55 and score <= 70:
                        score_level = "3"
                    elif score > 70 and score <= 85:
                        score_level = "4"
                    elif score > 85:
                        score_level = "5"

                result["tea_praise_score"] = int(tea_praise_score)
                result["tea_speak_speed_score"] = int(tea_speak_speed_score)
                result["stu_speak_score"] = int(stu_speak_score)
                result["total_score"] = int(score)
                result["score_level"] = score_level
                result["stu_speak_percent"] = round(stu_speak_percent, 2)
                result["stu_names"] = stu_names
                result["extra"]["stu_ids"] = stu_ids
                result["tea_total_praise_times"] = result["tea_praise_times"]+result["gesture_encourage_times"]
                result["tea_praise_level"] = tea_praise_level
                result["tea_speak_speed_level"] = tea_speak_speed_level
                result["stu_speak_level"] = stu_speak_level
                result["extra"]["tea_score_2"] = int(score_2)
        return results

    def cal_stu_score(self, results):  # 学生课堂打分
        for i, result in enumerate(results):
            if result["role"] == 1:  # 学生
                emo_score = result["emo_score"]  # 表情得分
                result["class_time"] = 3600 if result["class_time"] <= 0 else result["class_time"]
                stu_speak_percent = result["speak_time"]/result["class_time"]
                if stu_speak_percent >= 0.19:
                    stu_speak_score = 100
                else:
                    stu_speak_score = 526.32*stu_speak_percent
                stu_speak_score = 100 if stu_speak_score > 100 else stu_speak_score
                stu_speak_score = 0 if stu_speak_score < 0 else stu_speak_score
                stu_face_percent = result["extra"]["face_percent"]
                stu_face_score = stu_face_percent*100
                stu_face_score = 100 if stu_face_score > 100 else stu_face_score
                score = 0.3*emo_score+0.4*stu_speak_score+0.3*stu_face_score  # 三个指标打分
                results[i]["total_score"] = int(score)
        return results

    def get_classtype(self, room_id):  # 获取课程类型
        try:
            if r.status_code == 200:
                r = json.loads(r.text)
                if r['message'] == 'Success':
                    self.class_type = r['data']['courseEditionName']
                else:
                    print('获取课程类型出错！')
        except Exception as ex:
            print(ex)

    def highlight_extract(self, asr_stu, emotion_list_stu, facecount_list, pos_scores, mouse_list, jump_num):  # 计算精彩瞬片片段,jump_num用于视频与语音时间同步
        try:
            score_list_origin = []  # 得分列表-非正则化
            time_list = []  # 得分对应的时间段列表-防止被归一化

            if len(asr_stu) == 0:  # 没有检测到学生语音-bu
                return [], 0, 0, 0, False
            else:  # 检测到学生语音
                # age_stu = np.nanmean(age_list_stu[:num_half_class])  # 学生半节课平均年龄
                for i, item_stu in enumerate(asr_stu):  # 循环语音列表  asr_stu: [t1,t2,text]
                    if item_stu[0] < 180000 or item_stu[0] > 1000*(len(facecount_list)-300):  # 跳过前3分钟和后5分钟，去除家长干扰
                        continue
                    word_num = 0  # 片段讲话字数
                    start_time_ms = item_stu[0]  # 当前段开始时间ms
                    for j in range(100):  # 向后查找超30s
                        if i+j > len(asr_stu)-1:
                            break
                        else:
                            item = asr_stu[i+j]
                            end_time_ms = item[1]  # 结束时间ms
                            t1 = math.floor(item[0]/1000)  # 对应的开口时间s
                            t2 = math.ceil(item[1]/1000)  # 对应的开口时间s
                            if t2-t1 > 45 or end_time_ms-start_time_ms > 45000:  # 单段时间过长大于45秒，可能是课件声音
                                break

                            open_mouse_num = np.sum(np.array(mouse_list[t1-jump_num:t2-jump_num]) > 0.2)
                            # mouse = np.max(mouse_list[t1:t2])  # 计算讲话时间段平均开口度
                            mouse_std = np.nanstd(mouse_list[t1-jump_num:t2-jump_num])
                            valid_mouse_num = math.ceil((t2-t1)/5)

                            if open_mouse_num >= valid_mouse_num and mouse_std > 0.05:  # 有效开口
                                word_num += len(item[2])  # 讲话字数累加
                            else:
                                word_num += 0

                            if end_time_ms-start_time_ms > 30000:  # 单段大于30秒
                                # is_stu = self.is_student(start_time_ms, end_time_ms, age_list_stu, age_stu)
                                # if not is_stu:
                                #     break
                                t1 = math.floor(start_time_ms/1000)
                                t2 = math.ceil(end_time_ms/1000)
                                word_num_rate = word_num/(t2-t1)  # 每秒讲话字数
                                emo = np.mean(emotion_list_stu[t1-jump_num:t2-jump_num])  # 计算整个片段学生表情均值,无表情为0
                                if emo > 0:
                                    time_list.append([start_time_ms, end_time_ms])
                                    score_list_origin.append([word_num_rate, emo])  # 【单位时间字数，表情得分0-1】
                                break
            if len(score_list_origin) == 0:
                return [], 0, 0, 0, False  # 未检测到精彩片段

            score_list_norm = np.array(score_list_origin)  # 分数列表正则化 [word_num_pers,emo]
            speak_min = np.min(score_list_norm[:, 0])
            speak_max = np.max(score_list_norm[:, 0])
            score_list_norm[:, 0] = (score_list_norm[:, 0]-speak_min)/(speak_max-speak_min) if speak_max != speak_min else 0

            emo_min = np.min(score_list_norm[:, 1])
            emo_max = np.max(score_list_norm[:, 1])
            score_list_norm[:, 1] = (score_list_norm[:, 1]-emo_min)/(emo_max-emo_min) if emo_max != emo_min else 0
            # score_list = normalize(score_list_real, axis=0, norm='max')  # 分数列表正则化 [word_num,emo]
            score_list = score_list_norm.tolist()
            for i, item in enumerate(score_list):
                score = 0.3*item[0]+0.7*item[1]  # 讲话，表情
                s_time = time_list[i][0]  # ms
                e_time = time_list[i][1]  # ms
                t1 = math.floor(s_time/1000)  # s
                t2 = math.ceil(e_time/1000)  # s
                have_other_people = (np.sum(np.array(facecount_list[t1-jump_num:t2-jump_num]) == 1) != t2-t1)
                if have_other_people:
                    item.append(0)
                else:
                    item.append(score)
                item.append(s_time)
                item.append(e_time)
                item.append(score_list_origin[i][0])  # 单位时间讲话字数
                item.append(score_list_origin[i][1])  # 真实表情得分0-1

            sorted_list = sorted(score_list, key=lambda x: x[2], reverse=True)  # [word_num_norm,emo_norm,score_norm,t1,t2,word_num,emo]
            # 选前四个，没有重叠的片段
            top_list = []  # 前四个片段列表 [word_num,emo,score_norm,t1,t2,word_num,emo]
            history_time = []
            for item in sorted_list:  # 找出前四个不重叠的片段 [word_num,emo,score_norm,t1,t2,word_num,emo]
                s_time = item[3]  # ms
                e_time = item[4]  # ms
                is_inhistory = False
                for t in history_time:  # 判断是否与历史片段重叠
                    if t[0]-e_time > 3000 or s_time-t[1] > 3000:
                        continue
                    else:
                        is_inhistory = True
                        break
                if is_inhistory:
                    continue
                else:
                    top_list.append(item)
                    history_time.append([s_time, e_time])
                if len(top_list) == 1:  # 片段数量1
                    break
            top_list = sorted(top_list, key=lambda x: x[3], reverse=False)  # 按时间顺序排列 [word_num,emo,score_norm,t1,t2,word_num,emo]
            # 计算精彩得分
            score_list = np.array(top_list)
            word_num = np.nanmean(score_list[:, 5])
            emo_score = np.nanmean(score_list[:, 6])
            speak_score = 50*word_num
            speak_score = 100 if speak_score > 100 else speak_score
            highlight_score = round(emo_score*0.7+speak_score*0.3, 2)
            highlight_score = 100 if highlight_score > 100 else highlight_score
            create_hightlight = False
            if emo_score >= 65 and speak_score >= 50:
                create_hightlight = True
            return top_list, highlight_score, round(emo_score, 2), round(speak_score, 2), create_hightlight
        except Exception as ex:
            print("精彩瞬间分析出错:"+str(ex))
            traceback.print_exc()
            return [], -1, 0, 0, False

    def get_highlight_url_composed(self, room_id, user_id, highlight_list, mp4_file_stu, highlight_file, interval_offset):  # 上下拼接视频
        try:
            if len(highlight_list) == 0:
                print('{0} 精彩瞬间分析结果为空'.format(room_id))
                return "", None
            tmp_file_list = []
            clip_list_stu = []
            clip_list_tea = []
            for i, item in enumerate(highlight_list):  # 学生每个片段截取
                output_file_name_stu = os.path.join(os.path.dirname(__file__), 'tmp/highlights/{0}_{1}_stu.mp4'.format(room_id, i))
                output_file_name_tea = os.path.join(os.path.dirname(__file__), 'tmp/highlights/{0}_{1}_tea.mp4'.format(room_id, i))
                tmp_file_list.append(output_file_name_stu)
                tmp_file_list.append(output_file_name_tea)
                start_sec, end_sec = math.floor(item[3]/1000)-1, math.ceil(item[4]/1000)+1

                self.cut_video(mp4_file_stu, output_file_name_stu, start_sec, end_sec)
                self.cut_video(self.tea_mp4_file, output_file_name_tea, start_sec-interval_offset, end_sec-interval_offset)

                # 读取本地
                clip_tea = VideoFileClip(output_file_name_tea).subclip(0, -1)
                clip_stu = VideoFileClip(output_file_name_stu).subclip(0, -1)
                if clip_tea.duration > clip_stu.duration:
                    clip_tea = clip_tea.subclip(0, clip_stu.duration)
                elif clip_tea.duration < clip_stu.duration:
                    clip_stu = clip_stu.subclip(0, clip_tea.duration)

                if i != 0:
                    clip_tea = clip_tea.fadein(1, (1, 1, 1))
                    clip_stu = clip_stu.fadein(1, (1, 1, 1))
                if i != len(highlight_list)-1:
                    clip_tea = clip_tea.fadeout(1, (1, 1, 1))
                    clip_stu = clip_stu.fadeout(1, (1, 1, 1))
                clip_list_stu.append(clip_stu)
                clip_list_tea.append(clip_tea)

            clip_stu_com = concatenate_videoclips(clip_list_stu)  # 合并视频
            clip_tea_com = concatenate_videoclips(clip_list_tea)  # 合并视频

            back_mask_tea = ImageClip('src/back_tea_0227.jpg', ismask=True)
            back_mask_stu = ImageClip('src/back_stu_0227.jpg', ismask=True)

            clip_img = ImageClip("src/back_0227.jpg").set_duration(clip_stu_com.duration)  # .set_fps(15)
            clip_stu_com = clip_stu_com.resize(width=320).margin(top=180, left=22, opacity=0).set_mask(back_mask_stu)
            clip_tea_com = clip_tea_com.resize(width=170).margin(top=523, left=177, opacity=0).set_mask(back_mask_tea)

            clip_video = CompositeVideoClip([clip_img, clip_stu_com, clip_tea_com])
            highlight_file_tmp = highlight_file.replace(".mp4", "_tmp.mp4")

            clip_video.write_videofile(highlight_file_tmp, audio_codec='aac', write_logfile=False, logger=None, threads=1)  # 保存到本地

            text = 'ffmpeg -y -threads 1 -i {0} -vf "scale=2*trunc(iw/2):-2,setsar=1" -profile:v main -pix_fmt yuv420p -loglevel quiet {1}'.format(highlight_file_tmp, highlight_file)  # 转码
            res = os.system(text)



            tmp_file_list.append(highlight_file)
            tmp_file_list.append(highlight_file_tmp)
            for f in tmp_file_list:
                if os.path.exists(f):
                    os.remove(f)

            return url, file_id
        except Exception as ex:
            print('{0} 精彩瞬间视频生成失败:{1}'.format(room_id, str(ex)))
            return "", None  # url,file_id

    def get_highlight_url_metabase(self, room_id, user_id, highlight_list, mp4_file_stu, highlight_file, interval_offset):  # 上下拼接视频
        try:
            if len(highlight_list) == 0:
                print('{0} 精彩瞬间分析结果为空'.format(room_id))
                return "", None
            tmp_file_list = []
            # input_vid = ffmpeg.input(mp4_file)
            # input_vid_tea = ffmpeg.input(self.tea_mp4_file)
            clip_list_stu = []
            clip_list_tea = []
            for i, item in enumerate(highlight_list):  # 学生每个片段截取
                output_file_name_stu = os.path.join(os.path.dirname(__file__), 'tmp/highlights/{0}_{1}_stu.mp4'.format(room_id, i))
                output_file_name_tea = os.path.join(os.path.dirname(__file__), 'tmp/highlights/{0}_{1}_tea.mp4'.format(room_id, i))
                tmp_file_list.append(output_file_name_stu)
                tmp_file_list.append(output_file_name_tea)
                start_sec, end_sec = math.floor(item[3]/1000)-1, math.ceil(item[4]/1000)+1

                self.cut_video(mp4_file_stu, output_file_name_stu, start_sec, end_sec)
                self.cut_video(self.tea_mp4_file, output_file_name_tea, start_sec-interval_offset, end_sec-interval_offset)

                # 读取本地
                clip_tea = VideoFileClip(output_file_name_tea).subclip(0, -1)
                clip_stu = VideoFileClip(output_file_name_stu).subclip(0, -1)
                if clip_tea.duration > clip_stu.duration:
                    clip_tea = clip_tea.subclip(0, clip_stu.duration)
                elif clip_tea.duration < clip_stu.duration:
                    clip_stu = clip_stu.subclip(0, clip_tea.duration)

                if i != 0:
                    clip_tea = clip_tea.fadein(1, (1, 1, 1))
                    clip_stu = clip_stu.fadein(1, (1, 1, 1))
                if i != len(highlight_list)-1:
                    clip_tea = clip_tea.fadeout(1, (1, 1, 1))
                    clip_stu = clip_stu.fadeout(1, (1, 1, 1))
                clip_list_stu.append(clip_stu)
                clip_list_tea.append(clip_tea)

            clip_stu_com = concatenate_videoclips(clip_list_stu)  # 合并视频
            clip_tea_com = concatenate_videoclips(clip_list_tea)  # 合并视频

            back_mask_tea = ImageClip('src/back_tea_0227.jpg', ismask=True)
            back_mask_stu = ImageClip('src/back_stu_0227.jpg', ismask=True)

            clip_img = ImageClip("src/back_0227.jpg").set_duration(clip_stu_com.duration)  # .set_fps(15)
            clip_stu_com = clip_stu_com.resize(width=320).margin(top=180, left=22, opacity=0).set_mask(back_mask_stu)
            clip_tea_com = clip_tea_com.resize(width=170).margin(top=523, left=177, opacity=0).set_mask(back_mask_tea)

            clip_video = CompositeVideoClip([clip_img, clip_stu_com, clip_tea_com])

            highlight_file_tmp = highlight_file.replace(".mp4", "_tmp.mp4")
            clip_video.write_videofile(highlight_file_tmp, audio_codec='aac', write_logfile=False, logger=None)  # 保存到本地

            text = 'ffmpeg -y -threads 1 -i {0} -vf "scale=2*trunc(iw/2):-2,setsar=1" -profile:v main -pix_fmt yuv420p -loglevel quiet {1}'.format(highlight_file_tmp, highlight_file)  # 转码
            res = os.system(text)


            tmp_file_list.append(highlight_file)
            tmp_file_list.append(highlight_file_tmp)
            for f in tmp_file_list:
                if os.path.exists(f):
                    os.remove(f)

            return url, None
        except Exception as ex:
            print('{0} 精彩瞬间视频生成失败:{1}'.format(room_id, str(ex)))
            return "", None  # url,file_id

    def cut_video(self, video_path, out_video_path, start_time, end_time):  # start_time,end_time单位秒
        try:
            text = 'ffmpeg -i {0} -ss {1} -to {2} -c:a aac -b:a 98k {3} -loglevel quiet -y -threads 1'.format(video_path, start_time, end_time, out_video_path)
            res = os.system(text)
            return True
        except Exception as ex:
            print("视频截取失败：{0}".format(ex))
            return False

    def cal_lightness(self, frame_rgb):  # 计算图像亮度-标准=0.3
        lightness=0
        try:
            hsv_image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            lightness = hsv_image[:, :, 2].mean()
        except Exception as ex:
            print("亮度计算出错：{0}".format(str(ex)))
        return lightness


def worker_gpu(share_var, share_lock):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    num = 0
    while True:
        gpu_util = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        mem_used = meminfo.used/1024**2
        if int(mem_used) > 13500:
            print('@@@@@@@@@@@@显存占用{}M,开始清理...'.format(mem_used))
            share_lock.acquire()
            for i in range(len(share_var)):
                share_var[i] = 1
            share_lock.release()
        if gpu_util == 100:
            num += 1
        else:
            num = 0
        if num > 30:
            print("GPU使用异常，系统将在30秒后重启！")
            time.sleep(30)
            break
        else:
            time.sleep(10)


if __name__ == "__main__":
    num_workers = 4  # 进程数
    test_mode = platform.platform()[:5] == 'macOS'  # 测试模式
    if test_mode:  # 测试模式
        worker_local()  # 本地运行
    else:  # 生产模式
        share_lock = multiprocessing.Manager().Lock()
        share_var = multiprocessing.Manager().list()
        for i in range(num_workers):
            share_var.append(0)
        for i in range(num_workers):
            worker_process = multiprocessing.Process(target=worker, args=(i+1,share_var, share_lock))
            worker_process.daemon = True  # 随父进程一起退出
            worker_process.start()
        worker_process_gpu = multiprocessing.Process(target=worker_gpu, args=(share_var, share_lock))  # 监测gpu使用率
        worker_process_gpu.daemon = True
        worker_process_gpu.start()
        worker_process_gpu.join()
    print("主进程异常退出...")
