import os
import sys
import cv2
import time
import traceback
import numpy as np
# from age_utils import Age
from OpenEyeAna import OpenEyeDec
from HeadPoseCal import HeadPoseCal
from PeopleCountAna import PeopleCountAna
from retinaface_onnx import Retinaface_onnx
from facepoints_utils import FacepointsDetector
from FacialExpression import FacialExpressionDetector
sys.path.append(os.path.dirname(__file__))


class OneImageAna:
    def __init__(self):
        self.retinaface_detector = Retinaface_onnx()  # retinaface人脸检测
        self.facialdector = FacialExpressionDetector()  # 表情识别
        self.peoplecount_detector = PeopleCountAna()  # 人数识别
        self.facepointdetector = FacepointsDetector()  # 98个关键点检测
        self.openeye_detector = OpenEyeDec()  # 张眼闭眼检测
        # self.age_detector = Age()  # 年龄识别

    def getResult(self, frame_rgb, detect_eye=False, detect_emo=True, detect_keypoints=False,detect_age=False):
        result, face_count = [], 0
        try:
            bboxes = self.retinaface_detector.detect(frame_rgb)
            if len(bboxes) > 0:
                gray_ori = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                for face in bboxes:
                    h, w = abs(face[3]-face[1]), abs(face[2]-face[0])
                    if h > 20 or w > 20:
                        face_count += 1
                    elif h < 20 or w < 20:  # 面部太窄，不做后续分析
                        continue
                    face = [int(face[0]-w//10), int(face[1]-h//10), int(face[2]+w//10), int(face[3]+h//10)]  # 不放大矩形框开口有误差

                    obj = {}
                    obj["face"] = face
                    frame_face = gray_ori[face[1]:face[3], face[0]:face[2]]
                    if frame_face.shape[0] < 5 or frame_face.shape[1] < 5:  # 截完面部过小
                        continue

                    if detect_emo:
                        if face[0] < 3 or face[1] < 3 or face[2] > frame_rgb.shape[1]-3 or face[3] > frame_rgb.shape[0]-3:  # 面部贴边
                            obj["emo"] = None, None
                        else:
                            obj["emo"] = self.facialdector.detect(frame_face)  # 表情识别
                    if detect_keypoints:
                        obj["keypoints98"] = self.facepointdetector.detect(frame_rgb, face)  # 98个关键点
                        obj["mouse"], obj["mouse_dis"] = HeadPoseCal.getMousePose98(obj["keypoints98"])  # 开口度
                        obj["face_angle"]=HeadPoseCal.get_facepose(obj["keypoints98"])
                        obj["pos_score"] = HeadPoseCal.geEyePose98_Score(obj["keypoints98"], frame_rgb.shape[1], frame_rgb.shape[0])
                        # obj["eye"] = HeadPoseCal.getEyePose98(obj["keypoints98"])  # 眼睛张合度
                    if detect_eye:  # 张眼闭眼检测
                        frame_face_rgb = frame_rgb[face[1]:face[3], face[0]:face[2], :]
                        index_pred, softmax_value = self.openeye_detector.detect(frame_face_rgb)
                        obj["open_eye"] = True if index_pred == 1 else False
                    # if detect_age:  # 每隔4帧识别一次-仅学生
                    #     faceimg_age = frame_rgb[face[1]-20:face[3]+20, face[0]-20:face[2]+20]
                    #     obj["age"] =6
                        # obj["age"] = self.age_detector.detect(faceimg_age)
                        # print(obj["age"])
                    # else:
                    #     obj["age"] = np.nan
                    result.append(obj)
            if len(result) == 1:  # 单个人脸
                result = result[0]
            elif len(result) > 1:  # 多个面部筛选
                sorted_list = sorted(result, key=lambda item: abs((item["face"][0]-item["face"][2])*(item["face"][1]-item["face"][3])), reverse=True)
                result = sorted_list[0]
            if face_count != 1:  # 人数不为1，启用后验模型
                face_count = self.peoplecount_detector.detect(frame_rgb)
            result = result if len(result) > 0 else None
            return result, face_count
        except Exception as ex:
            traceback.print_exc()
            return None, 0
