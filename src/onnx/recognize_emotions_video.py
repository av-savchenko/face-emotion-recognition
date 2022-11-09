import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

def process_video(videofile=0):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    #face_mesh=mp_face_mesh.FaceMesh(max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    face_mesh=mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

    model_name='enet_b0_8_best_vgaf'
    #model_name='enet_b0_8_va_mtl'
    fer=HSEmotionRecognizer(model_name=model_name)

    maxlen=15 #51
    recent_scores=deque(maxlen=maxlen)

    cap = cv2.VideoCapture(videofile)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            #print("Ignoring empty camera frame.")
            break

        total_start = time.time()
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        results = face_mesh.process(image_rgb)
        elapsed = (time.time() - start)
        print('Face mesh elapsed:',elapsed)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if results.multi_face_landmarks:
            height,width,_=image.shape
            for face_landmarks in results.multi_face_landmarks:
                if False:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                x1 = y1 = 1
                x2 = y2 = 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = lm.x, lm.y
                    if cx<x1:
                        x1=cx
                    if cy<y1:
                        y1=cy
                    if cx>x2:
                        x2=cx
                    if cy>y2:
                        y2=cy
                if x1<0:
                    x1=0
                if y1<0:
                    y1=0
                x1,x2=int(x1*width),int(x2*width)
                y1,y2=int(y1*height),int(y2*height)
                face_img=image_rgb[y1:y2,x1:x2,:]
                if np.prod(face_img.shape)==0:
                    print('Empty face ', x1,x2,y1,y2)
                    continue
                
                start = time.time()
                emotion,scores=fer.predict_emotions(face_img,logits=True)
                elapsed = (time.time() - start)
                recent_scores.append(scores)

                scores=np.mean(recent_scores,axis=0)
                emotion=np.argmax(scores)
                print(scores,fer.idx_to_class[emotion], 'Emotion elapsed:',elapsed)
                
                cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
                fontScale=1
                min_y=y1
                if min_y<0:
                    min_y=10
                cv2.putText(image, fer.idx_to_class[emotion], (x1, min_y), cv2.FONT_HERSHEY_PLAIN , fontScale=fontScale, color=(0,255,0), thickness=1)
        else:
            recent_concentartion_indices.append(0)        
        elapsed = (time.time() - total_start)
        print('Total frame processing elapsed:',elapsed)
        cv2.imshow('Facial emotions', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break

    face_mesh.close()
    cap.release()

if __name__ == '__main__':
    if len(sys.argv)==2:
        process_video(sys.argv[1])
    else:
        process_video()