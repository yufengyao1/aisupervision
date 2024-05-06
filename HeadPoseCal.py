# 输入图像储存和面部68个关键点，计算头部姿势角度
import cv2
import math
import numpy as np

def get_num(point_dict, name, axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):
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

def point_line(point, line):
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

def point_point(point_1, point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return distance

class HeadPoseCal():
    @staticmethod
    def get_image_points_from_landmark_shape(landmark_shape):
        image_points = np.array([
            (landmark_shape.part(30).x, landmark_shape.part(30).y),
            (landmark_shape.part(8).x, landmark_shape.part(8).y),
            (landmark_shape.part(36).x, landmark_shape.part(36).y),
            (landmark_shape.part(45).x, landmark_shape.part(45).y),
            (landmark_shape.part(48).x, landmark_shape.part(48).y),
            (landmark_shape.part(54).x, landmark_shape.part(54).y)
        ], dtype="double")
        return image_points

    @staticmethod
    def get_pose_estimation(img_size, image_points):
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                      image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

    @staticmethod
    def get_euler_angle(rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)
        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta
        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        pitch = math.atan2(t0, t1)
        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)
        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)
        return 0, Y, X, Z

    @staticmethod
    def getHeadPoseAngle(picsize, facepoints):  # 头部角度计算
        facepoints = HeadPoseCal.get_image_points_from_landmark_shape(facepoints)
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = HeadPoseCal.get_pose_estimation(picsize, facepoints)
        # ret, pitch, yaw, roll = HeadPoseCal.get_euler_angle(rotation_vector)
        # pitch=180-(360+pitch)%360
        # if(abs(roll)>90):
        #     roll = (180-abs(roll)) if roll > 0 else (abs(roll)-180)
        yaw, pitch, roll = 0, 0, 0
        # 计算鼻头朝向
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 340.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)  # 340表示图像尺寸
        return (yaw, pitch, roll, nose_end_point2D)

    @staticmethod
    def getEyePose(facepoints):  # 眼睛闭合度计算
        t1 = math.hypot(facepoints.part(37).x-facepoints.part(41).x, facepoints.part(37).y-facepoints.part(41).y)
        t2 = math.hypot(facepoints.part(38).x-facepoints.part(40).x, facepoints.part(38).y-facepoints.part(40).y)
        t3 = math.hypot(facepoints.part(36).x-facepoints.part(39).x, facepoints.part(36).y-facepoints.part(39).y)
        r1 = (t1+t2)/t3/2
        t4 = math.hypot(facepoints.part(43).x-facepoints.part(47).x, facepoints.part(43).y-facepoints.part(47).y)
        t5 = math.hypot(facepoints.part(44).x-facepoints.part(46).x, facepoints.part(44).y-facepoints.part(46).y)
        t6 = math.hypot(facepoints.part(42).x-facepoints.part(45).x, facepoints.part(42).y-facepoints.part(45).y)
        r2 = (t4+t5)/t6/2
        return round((r1+r2)/2, 2)

    @staticmethod
    def getMousePose(facepoints):  # 嘴巴闭合度计算
        # 内嘴角闭合度
        t1 = math.hypot(facepoints.part(61).x-facepoints.part(67).x, facepoints.part(61).y-facepoints.part(67).y)
        t2 = math.hypot(facepoints.part(63).x-facepoints.part(65).x, facepoints.part(63).y-facepoints.part(65).y)
        t3 = math.hypot(facepoints.part(60).x-facepoints.part(64).x, facepoints.part(60).y-facepoints.part(64).y)
        distance = math.hypot(facepoints.part(57).x-facepoints.part(51).x, facepoints.part(57).y-facepoints.part(51).y)
        return round((t1+t2)/t3/2, 2), int(distance)

    def get_image_points98(landmark_shape):  # 98点抽取6个关键点
        image_points = np.array([
            (landmark_shape[54, 0], landmark_shape[54, 1]),
            (landmark_shape[16, 0], landmark_shape[16, 1]),
            (landmark_shape[60, 0], landmark_shape[60, 1]),
            (landmark_shape[72, 0], landmark_shape[72, 1]),
            (landmark_shape[76, 0], landmark_shape[76, 1]),
            (landmark_shape[82, 0], landmark_shape[82, 1])
        ], dtype="double")
        return image_points

    def getHeadPoseAngle98(picsize, facepoints):  # 头部角度计算-98个点
        facepoints = HeadPoseCal.get_image_points98(facepoints)  # 6个关键点
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = HeadPoseCal.get_pose_estimation(picsize, facepoints)  # 基于6个点姿态估计
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 340.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)  # 计算视线点朝向，340表示图像尺寸
        return nose_end_point2D

    def getMousePose98(facepoints):  # 开口度计算-98个点
        # 内嘴角闭合度
        t1 = math.hypot(facepoints[90][0]-facepoints[94][0], facepoints[90][1]-facepoints[94][1])
        t2 = math.hypot(facepoints[88][0]-facepoints[92][0], facepoints[88][1]-facepoints[92][1])
        return t1/t2, t2

    def getEyePose98(facepoints):  # 眼睛闭合度计算
        t1 = math.hypot(facepoints[61][0] - facepoints[67][0], facepoints[61][1] - facepoints[67][1])
        t2 = math.hypot(facepoints[63][0] - facepoints[65][0], facepoints[63][1] - facepoints[65][1])
        t3 = math.hypot(facepoints[60][0] - facepoints[64][0], facepoints[60][1] - facepoints[64][1])
        r1 = (t1 + t2) / t3 / 2
        t4 = math.hypot(facepoints[69][0] - facepoints[75][0], facepoints[69][1] - facepoints[75][1])
        t5 = math.hypot(facepoints[71][0] - facepoints[73][0], facepoints[71][1] - facepoints[73][1])
        t6 = math.hypot(facepoints[68][0] - facepoints[72][0], facepoints[68][1] - facepoints[72][1])
        r2 = (t4 + t5) / t6 / 2
        eye = max(r1, r2)
        return round(eye, 2)

    def geEyePose98_Score(facepoints, frame_w, frame_h):  # 计算眼睛位置得分
        x_ratio = facepoints[51][0]/frame_w
        score_x = 200*x_ratio if x_ratio < 0.5 else 200-200*x_ratio
        score_x = 100 if score_x > 100 else score_x
        score_x = 0 if score_x < 0 else score_x

        y_ratio = facepoints[51][1]/frame_h
        if y_ratio >= 0.35 and y_ratio <= 0.4:
            score_y = 100
        else:
            score_y = 285.7*y_ratio if y_ratio < 0.35 else 200-250*y_ratio
        score_y = 100 if score_y > 100 else score_y
        score_y = 0 if score_y < 0 else score_y

        score1 = min(score_x,score_y)

        dis = math.hypot(facepoints[0][0] - facepoints[32][0], facepoints[0][1] - facepoints[32][1])
        ratio = dis/frame_w
        if ratio > 0.25 and ratio < 0.4:
            score = 100
        else:
            score = 666.7*ratio-66.67 if ratio < 0.25 else 200-250*ratio
        score2 = 100 if score > 100 else score
        score2 = 0 if score < 0 else score
        return int(min(score1,score2))

    @staticmethod
    def get_facepose(pre_landmark): #简单方式计算头部姿态
        yaw,pitch,roll=0,0,0
        try:
            i = 0
            point_dict = {}
            for (x, y) in pre_landmark.astype(np.float32):
                point_dict[f'{i}'] = [x, y]
                i += 1
            # yaw
            point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
            point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
            point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
            crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
            yaw_mean = point_point(point1, point31) / 2
            yaw_right = point_point(point1, crossover51)
            yaw = (yaw_mean - yaw_right) / yaw_mean
            yaw = int(yaw * 71.58 + 0.7037)

            # pitch
            pitch_dis = point_point(point51, crossover51)
            if point51[1] < crossover51[1]:
                pitch_dis = -pitch_dis
            pitch = int(1.497 * pitch_dis + 18.97)

            # roll
            roll_tan = abs(get_num(point_dict, 60, 1) - get_num(point_dict, 72, 1)) / abs(get_num(point_dict, 60, 0) - get_num(point_dict, 72, 0))
            roll = math.atan(roll_tan)
            roll = math.degrees(roll)
            if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
                roll = -roll
            roll = int(roll)
            return yaw,pitch,roll
        except:
            return yaw,pitch,roll