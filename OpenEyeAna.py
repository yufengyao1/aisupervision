import os
import cv2
import numpy as np
import onnxruntime


class OpenEyeDec:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4  # 设置线程数
        opts.inter_op_num_threads = 4  # 设置parallel线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), 'weights', 'eye_mobilenetv2_1019.onnx'), opts, providers=provider)  # ONNX模式检测
        self.input_name = self.onnx_session.get_inputs()[0].name

    def detect(self, frame):  # rgb格式frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
        frame = frame/255
        frame = (frame-0.5)/0.5  # 转换到-1，1之间
        frame = frame.transpose(2, 0, 1)
        inputs = np.expand_dims(frame, axis=0)
        inputs = inputs.astype(np.float32)
        inputs = {self.input_name: inputs}
        pred = self.onnx_session.run(None, inputs)

        pred = np.squeeze(pred)
        index_pred = np.argmax(pred)
        # pred = self.cal_softmax(pred)
        # softmax_value = round(pred[index_pred], 2)
        return index_pred, 1

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
