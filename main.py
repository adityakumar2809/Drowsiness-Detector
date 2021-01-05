import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance


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


def main():
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('cam_screen')

    while True:
        ret, frame = cam.read()

        if not ret:
            print('Failed to grab the image')
            break

        faces = detector(frame, 0)

        for face in faces:
            shape = predictor(frame, face)
            shape = face_utils.shape_to_np(shape)

            eye = calculateEAR(shape)

        # for (i, face) in enumerate(faces):
        #     shape = predictor(frame, face)
        #     shape = face_utils.shape_to_np(shape)

        #     for (x, y) in shape:
        #         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow('cam_screen', frame)

        k = cv2.waitKey(1)
        if k == 27:
            print('ESC Key Pressed. Closing window...')
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
