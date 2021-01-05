import cv2
import dlib
import pyttsx3
import numpy as np
from imutils import face_utils
from scipy.spatial import distance
from threading import Thread

import constants


def alertUser(message):
    global currently_speaking
    global engine

    engine.say(message)
    engine.runAndWait()
    currently_speaking = False


def eyeAspectRatio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def calculateEAR(shape):
    (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    left_eye = shape[left_start: left_end]
    right_eye = shape[right_start: right_end]

    left_EAR = eyeAspectRatio(left_eye)
    right_EAR = eyeAspectRatio(right_eye)

    average_EAR = (left_EAR + right_EAR) / 2.0

    return (average_EAR, left_eye, right_eye)


def calculateLipDistance(shape):
    top_lip = shape[50: 53]
    top_lip = np.concatenate((top_lip, shape[61: 64]))

    bottom_lip = shape[56: 59]
    bottom_lip = np.concatenate((bottom_lip, shape[65: 68]))

    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)

    distance = abs(top_mean[1] - bottom_mean[1])
    return distance


def main():
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('cam_screen')

    low_ear_frame_counter = 0
    global currently_speaking

    while True:
        ret, frame = cam.read()

        if not ret:
            print('Failed to grab the image')
            break

        faces = detector(frame, 0)

        for face in faces:
            shape = predictor(frame, face)
            shape = face_utils.shape_to_np(shape)

            eye_details = calculateEAR(shape)
            average_EAR = eye_details[0]
            left_eye = eye_details[1]
            right_eye = eye_details[2]

            lip_distance = calculateLipDistance(shape)

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            lip_shape = shape[48: 60]
            cv2.drawContours(frame, [lip_shape], -1, (0, 255, 0), 1)

            if average_EAR < constants.AVERAGE_EAR_THRESHOLD:
                low_ear_frame_counter += 1
                if low_ear_frame_counter > \
                        constants.LOW_EAR_CONSECUTIVE_FRAMES:
                    if not currently_speaking:
                        currently_speaking = True
                        t = Thread(
                            target=alertUser,
                            args=('Please wake up master!',)
                        )
                        t.daemon = True
                        t.start()
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            else:
                low_ear_frame_counter = 0

            if lip_distance > constants.LIP_DISTANCE_THRESHOLD:
                if not currently_speaking:
                    currently_speaking = True
                    t = Thread(
                        target=alertUser,
                        args=('Not a good time to sleep master!',)
                    )
                    t.daemon = True
                    t.start()
                cv2.putText(
                    frame,
                    "YAWN ALERT!",
                    (10, 60),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        cv2.imshow('cam_screen', frame)

        k = cv2.waitKey(1)
        if k == 27:
            print('ESC Key Pressed. Closing window...')
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    currently_speaking = False
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    main()
