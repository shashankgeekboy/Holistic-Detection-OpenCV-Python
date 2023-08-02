import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic()
mpDraw = mp.solutions.drawing_utils
drawing_specs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    if not success:
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(imgRGB)

    mpDraw.draw_landmarks(img, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_specs, drawing_specs)
    mpDraw.draw_landmarks(img, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_specs, drawing_specs)
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS, drawing_specs, drawing_specs)

    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            cv2.circle(img, (x, y), 1, (0, 255, 0), 1)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
