import os
import cv2
import onnxruntime
import numpy as np


class HandDec:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4  # 设置线程数
        opts.inter_op_num_threads = 4  # 设置parallel线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        # opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # provider = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights', 'tea_gesture_mobilenetv2_1206.onnx'), opts, providers=provider)  # ONNX模式检测
        self.input_name = self.onnx_session.get_inputs()[0].name

    def cal_softmax(self, x):
        x = np.array(x)
        x = np.exp(x)
        x.astype('float32')
        if x.ndim == 1:
            sumcol = sum(x)
            for i in range(x.size):
                x[i] = x[i]/float(sumcol)
        if x.ndim > 1:
            sumcol = x.sum(axis=0)
            for row in x:
                for i in range(row.size):
                    row[i] = row[i]/float(sumcol[i])
        return x

    def detect(self, frame):  # rgb格式frame
        # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #转换成rgb
        frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
        # frame = frame[0:115, 0:155]
        # frame=frame[-175:,0:235]
        frame = frame/255
        frame = (frame-0.5)/0.5  # 转换到-1，1之间
        frame = frame.transpose(2, 0, 1)
        inputs = np.expand_dims(frame, axis=0)
        inputs = inputs.astype(np.float32)
        inputs = {self.input_name: inputs}
        pred = self.onnx_session.run(None, inputs)
        pred = np.squeeze(pred)
        index_pred = np.argmax(pred)
        return "", index_pred
        # if index_pred==1:
        #     return "good",index_pred
        # else:
        #     return None,0
        # if pred[1]>pred[0]:
        #     return "good"
        # else:
        #     return None
        # pred = self.cal_softmax(pred)
        # if pred[1]>0.8:
        #     return "good"
        # else:
        #     return None
