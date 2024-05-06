import numpy as np
import os
import cv2
import math
import onnxruntime


class FacepointsDetector:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4  # 设置线程数
        opts.inter_op_num_threads = 4  # 设置parallel线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        # opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # provider = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights', 'facepoints98_lite_sim.onnx'), opts,providers=provider)  # ONNX模式检测
        self.input_name = self.onnx_session.get_inputs()[0].name

    def detect(self, frame, face): #rgb格式
        try:
            # onnx模型推理面部98个点
            face_img = frame[face[1]:face[3], face[0]:face[2], :]
            w, h = face[2]-face[0], face[3]-face[1]
            max_size = np.max([w, h])
            # new_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            new_face_img = cv2.copyMakeBorder(face_img, int((max_size-h)/2), int((max_size-h)/2), int((max_size-w)/2), int((max_size-w)/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 图像边缘扩展
            scale = new_face_img.shape[0]/112
            new_face_img = cv2.resize(new_face_img, (112, 112))
            new_face_img = new_face_img/255

            inputs = np.transpose(new_face_img, (2, 0, 1))
            inputs = np.expand_dims(inputs, axis=0)
            inputs = inputs.astype(np.float32)
            inputs = {self.input_name: inputs}
            pred = self.onnx_session.run(None, inputs, None)
            pred = pred[0].reshape(-1, 2) * [112, 112]
            for p in pred:
                p[0] = face[0]-int((max_size-w)/2)+p[0]*scale
                p[1] = face[1]-int((max_size-h)/2)+p[1]*scale
            return pred
        except Exception as ex:
            print('emotion ana error:'+str(ex))
            return []

    def get_angles(self, point_dict):
        # yaw
        point1 = [point_dict[1][0], point_dict[1][1]]
        point31 = [point_dict[31][0], point_dict[31][1]]
        point51 = [point_dict[51][0], point_dict[51][1]]
        crossover51 = self.point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = self.point_point(point1, point31) / 2
        yaw_right = self.point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = int(yaw * 71.58 + 0.7037)

        # pitch
        pitch_dis = self.point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = int(1.497 * pitch_dis + 18.97)

        # roll
        roll_tan = abs(point_dict[60][1] - point_dict[72][1]) / abs(point_dict[60][0] - point_dict[72][0])
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        return yaw,pitch,roll

    def cross_point(self, line1, line2):
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]

        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]

        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        if (x4 - x3) == 0:
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)
            b2 = y3 * 1.0 - x3 * k2 * 1.0
        if k2 == None:
            x = x3
        else:
            x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        return [x, y]

    def point_line(self, point, line):
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        x3 = point[0]
        y3 = point[1]

        k1 = (y2 - y1)*1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        k2 = -1.0/k1
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        return [x, y]

    def point_point(self, point_1, point_2):
        x1 = point_1[0]
        y1 = point_1[1]
        x2 = point_2[0]
        y2 = point_2[1]
        distance = ((x1-x2)**2 + (y1-y2)**2)**0.5
        return distance
