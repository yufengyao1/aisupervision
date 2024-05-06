import os
import cv2
import onnxruntime
import numpy as np


class PeopleCountAna:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4  # 设置线程数
        opts.inter_op_num_threads = 4  # 设置parallel线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights', 'peoplecount_mobilenetv2_112_112.onnx'), opts, providers=provider)  # ONNX模式检测
        self.input_name=self.onnx_session.get_inputs()[0].name

    def detect(self, frame): #rgb格式frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换成rgb
        frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
        frame = frame/255
        frame = (frame-0.5)/0.5  # 转换到-1，1之间
        frame = frame.transpose(2, 0, 1)
        inputs = np.expand_dims(frame, axis=0)
        inputs = inputs.astype(np.float32)
        inputs = {self.input_name: inputs}
        pred = self.onnx_session.run(None, inputs)
        pred = np.squeeze(pred)
        max_index = np.argmax(pred)
        return max_index
