import cv2
import sys
import numpy as np
import pyautogui as gui
import mediapipe as mp
from threading import Thread


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
cam = cv2.VideoCapture(0)
mx,my = 0,0
draw = True

def mouse_thread():
    print("SUB PROCESSSS")
    if not draw:
        while cam.isOpened():
            gui.moveRel(np.floor(250*mx),np.floor(250*my),0.1,gui.linear,False,False)

def detect_arm(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2

    if not draw:
        if left_wrist.x < left_shoulder.x and left_wrist.x > right_shoulder.x:
            gui.keyDown('W')
        else:
            gui.keyUp('W')

    if left_wrist.y < shoulder_avg_y and right_wrist.y < shoulder_avg_y:
        if not draw:
            gui.keyDown('space')
            gui.sleep(0.5)
            gui.keyUp('space')
        return "JMP"
    elif left_wrist.y < shoulder_avg_y:
        if not draw:
            gui.leftClick()
        return "Left Click"
    elif right_wrist.y < shoulder_avg_y:
        if not draw:
            gui.rightClick()
        return "Right Click"
    else:
        return "Neutral"

def detect_neck_rotation(landmarks):
    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_midpoint_y = (left_shoulder.y + right_shoulder.y) / 2

    lr = (nose.x - shoulder_midpoint_x)/(left_shoulder.x - right_shoulder.x)
    ud =-(nose.y - shoulder_midpoint_y)/(left_shoulder.x - right_shoulder.x)

    global mx,my
    if lr <  -0.1:
        mx = -lr
    elif lr > 0.1:
        mx = -lr
    else:
        mx = 0

    if ud < 0.5:    
        my = -ud+0.675
    elif ud > 0.85:
        my = -ud+0.675
    else:
        my = 0


if __name__ == "__main__":

    if len(sys.argv) == 2:
        draw = False   

    thread1 = Thread(target=mouse_thread,args=())
    thread1.start()
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        results = pose.process(frame)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if results.pose_landmarks:
            detect_neck_rotation(results.pose_landmarks.landmark)
            hand_direction = detect_arm(results.pose_landmarks.landmark)
            
            if draw:
                cv2.putText(
                    frame,
                    f'Lean: {mx} {my}', 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f'Hand: {hand_direction}', 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 100), 2, cv2.LINE_AA
                )

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 250), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 0, 0), thickness=2, circle_radius=2)
                )

        if draw:
            cv2.imshow('Pose Estimation', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    thread1.join()
    cam.release()
    cv2.destroyAllWindows()
    pose.close()
