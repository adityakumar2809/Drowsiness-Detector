import cv2
import dlib
from imutils import face_utils


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

        cv2.imshow('cam_screen', frame)

        k = cv2.waitKey(1)
        if k == 27:
            print('ESC Key Pressed. Closing window...')
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
