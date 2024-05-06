import cv2


class Age:
    def __init__(self):
        ageProto = "weights/opencv/age_deploy.prototxt"
        ageModel = "weights/opencv/age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = [[0, 2], [4, 6], [8, 12], [15, 20], [25, 32], [38, 43], [48, 53], [60, 100]]

    def detect(self, frame_face):
        try:
            blob = cv2.dnn.blobFromImage(frame_face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=True)
            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]
            return age[1]
        except:
            return 6
