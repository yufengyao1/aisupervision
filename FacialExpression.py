import numpy as np
import os
import cv2
import onnxruntime
class FacialExpressionDetector:
    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4 #设置线程数
        opts.inter_op_num_threads = 4 #设置parallel线程数
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session=onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__),'weights', 'facialexpression.onnx'),opts,providers=provider) ##ONNX模式检测
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.input_name = self.onnx_session.get_inputs()[0].name
    def detect(self,gray):    
        try:
            gray=cv2.resize(gray, (44, 44), interpolation=cv2.INTER_AREA)
            img = gray[:, :, np.newaxis]
            img=img/255
            img = np.concatenate((img, img, img), axis=2)
            img=img.transpose(2,0,1)
            inputs=np.expand_dims(img, axis=0)
            inputs=inputs.astype(np.float32)
            inputs={self.input_name:inputs}
            outs=self.onnx_session.run(None,inputs)
            outputs=np.array(outs[0])
            x=outputs[0]
            # ###求softmax
            x = x - x.max(axis=None, keepdims=True)
            y = np.exp(x)
            score= y / y.sum(axis=None, keepdims=True)
            predicted=np.argmax(score)
            emojis=self.class_names[predicted]
            score=score.tolist()
            return [emojis,score]
        except Exception as ex:
            print('emotion ana error:'+str(ex))
            return []