import cv2


def main():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('screen')

    while True:
        ret, frame = cam.read()

        if not ret:
            print('Failed to grab the image')
            break

        cv2.imshow('screen', frame)

        k = cv2.waitKey(1)
        if k == 27:
            print('ESC Key Pressed. Closing window...')
            break

    cam.release()
    cv2.distroyAllWindows()


if __name__ == "__main__":
    main()
