import imutils
import dlib
import cv2
import pygame

from scipy.spatial import distance
from imutils import face_utils


def play_sound(path):
    pygame.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.14
frame_check = 5
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("./model.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(1)
flag = 0
sound_flag = False

while True:

    ret, frame=cap.read()
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:

        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        print(ear)

        if ear > thresh:
            flag += 1
            if flag >= frame_check:
                
                if not sound_flag:
                    play_sound("./alarm.mp3")
                    sound_flag = True

                cv2.putText(frame, "********************************** TRAMPOSO **********************************", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "********************************** TRAMPOSO **********************************", (10, 800),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0
            sound_flag = False

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()
