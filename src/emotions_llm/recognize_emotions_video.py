import sys
import time
import threading
import time
import subprocess
import os
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import pyaudio
import wave
import speech_recognition as sr

import textwrap 

from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

stopEvent = threading.Event()
recognized_text=''
ai_answer=''
recognized_emotion=''


#===============
emotions=["happy","sad","fear","angry"]
colors=[(0,128,0),(128,0,0),(128, 0, 128),(0,0,255)] #bgr

emotion_prompt="What is the {} one-paragraph response to "
additional_emotion_prompt="What is the {} one-paragraph response to the initial question?"
aggregation_prompt="Aggregate the answers to form a final response "
user_emotion_prompt=". Take into account that I am {} now."

#question2="suggestion of implementing emotional answers in a personalized assistant"

def get_emotion_prompt(emotion,prompt):
    return emotion_prompt.format(emotion)+prompt

def get_additional_emotion_prompt(emotion):
    return additional_emotion_prompt.format(emotion)

def get_aggregation_prompt(user_emotion):
    return aggregation_prompt if user_emotion=='neutral' else aggregation_prompt+user_emotion_prompt.format(user_emotion)

def process_multiple_emotional_agents2(question,user_emotion='neutral', delay=5):
    global ai_answer
    messages=[]
    for i,emotion in enumerate(emotions):
        if stopEvent.is_set():
            return
        if i==0:
            messages.append({
              "role": "user",
              "content": get_emotion_prompt(emotion,question)
            })
        else:
            messages.append({
              "role": "user",
              "content": get_additional_emotion_prompt(emotion)
            })
        assistant=get_completion_from_messages(messages)
        messages.append({
          "role": "assistant",
          "content": assistant
        })
        print(emotion,assistant,"\n\n")
        ai_answer=emotion+": "+assistant
        time.sleep(delay)
        
    messages.append({
      "role": "user",
      "content": get_aggregation_prompt(user_emotion)
    })
    assistant=get_completion_from_messages(messages)
    print('Summary:',assistant)
    ai_answer="Summary (user emotion "+user_emotion+"): "+assistant

import nest_asyncio
nest_asyncio.apply()
import g4f
#model_name="gpt-4-0613"
model_name="gpt-3.5-turbo"
#model_name='text-davinci-003'
#model_name=g4f.models.default

def get_completion(content):
    response = g4f.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        provider=g4f.Provider.Aura
    )
    
    return response

def get_completion_from_messages(messages):
    response = g4f.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
        provider=g4f.Provider.Aura
    )
    return response

#===============


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#face_mesh=mp_face_mesh.FaceMesh(max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
face_mesh=mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

#model_name='enet_b0_8_best_vgaf'
model_name='enet_b0_8_va_mtl'
fer=HSEmotionRecognizer(model_name=model_name)
emotion_idx_to_class={0: 'angry', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}

def draw_recognized_text(img, title, text, start_y=0, img_height=-1):
    height,width,_=img.shape
    font=cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_thickness = 1
    font_size = 1
    title_size = cv2.getTextSize(title, font, font_size, font_thickness)[0]
    if img_height==-1:
        img_height=img.shape[1]
    #print(start_y,title_size)
    #cv2.putText(img, title, (0, title_size[1]), font, fontScale=font_size, color=(0,0,0), thickness=font_thickness)
    cv2.putText(img, title, (int((img_height - title_size[0]) / 2), title_size[1]+start_y+2), font, fontScale=font_size, color=(0,0,0), thickness=font_thickness)

    font_size = 0.5
    textsize = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    max_width=92 #width//textsize[1]
    #print(textsize,len(text),width,textsize[1],width//textsize[1],len(recognized_text)//max(1,width//textsize[1]))
    wrapped_text = textwrap.wrap(text, width=max_width)
    x=gap=0
    y=title_size[1]+start_y
    color=(0,0,0)
    for i, emotion in enumerate(emotions):
        if text.startswith(emotion):
            color=colors[i]
            print(text,emotion,color)
            break
        
    for i, line in enumerate(wrapped_text):
        #textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        gap = textsize[1] + 10
    
        y = int((2*(title_size[1]+start_y) + textsize[1]) / 2) + (i+1) * gap
        #x = 0 #int((img.shape[1] - textsize[0]) / 2)
        #print(line,len(line),x,y)
        
    
        cv2.putText(img, line, (x, y), font,
                    font_size, 
                    color, 
                    font_thickness, 
                    lineType = cv2.LINE_AA)
    #print('end',y,gap)
    return y#+gap
        
def process_video(videofile=0):
    global recognized_text,ai_answer,recognized_emotion
    START_HEIGHT=640
    maxlen=15#15,51
    recent_scores=deque(maxlen=maxlen)

    cv2.namedWindow('Emotions', cv2.WINDOW_NORMAL)
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
        #print(image.shape)
        large_image=255*np.ones((360,START_HEIGHT+300,3), np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height,width,_=image.shape

        start = time.time()
        results = face_mesh.process(image_rgb)
        elapsed = (time.time() - start)
        #print('Face mesh elapsed:',elapsed)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if True:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
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
                #print(scores,fer.idx_to_class[emotion], 'Emotion elapsed:',elapsed)
                
                #cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
                if True:
                    face_height,face_width=y2-y1,x2-x1
                    y1=max(0,y1-face_height//3)
                    y2=min(height,y2+face_height//5)
                    x1=max(0,x1-face_width//5)
                    x2=min(width,x2+face_width//5)
                    image=image[y1:y2,x1:x2,:]
                    image=cv2.resize(image,(300,360))
                    recognized_emotion=emotion_idx_to_class[emotion]
                    color=(0,0,0)
                    for e,c in zip(emotions,colors):
                        if e==recognized_emotion:
                            color=c
                            break
                    cv2.putText(image, recognized_emotion, (10, 20), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=2)
                    break
                min_y=y1
                if min_y<0:
                    min_y=10
                #cv2.putText(image, fer.idx_to_class[emotion], (x1, min_y), cv2.FONT_HERSHEY_PLAIN , fontScale=1, color=(0,255,0), thickness=1)
        elapsed = (time.time() - total_start)
        #print('Total frame processing elapsed:',elapsed)
        #print(image.shape,large_image.shape,width,height)
        image=cv2.resize(image,(300,360))
        large_image[:,START_HEIGHT:,:]=image
        if recognized_text!='':
            y_shift=draw_recognized_text(large_image,'Recognized text',recognized_text,0,START_HEIGHT)
            if ai_answer!='':
                draw_recognized_text(large_image,'AI answer',ai_answer,y_shift,START_HEIGHT)
        cv2.imshow('Emotions', large_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    face_mesh.close()
    cap.release()
    stopEvent.set()


def process_audio():
    global recognized_text,ai_answer,recognized_emotion
    r = sr.Recognizer()
    print("Ready to capture audio")
    with sr.Microphone() as source:
        while not stopEvent.is_set():
            # read the audio data from the default microphone
            audio_data = r.record(source, duration=10)
            print("Recognizing...")
            try:
                recognized_text = r.recognize_google(audio_data)#,language='ru')
                print(recognized_text)
                if stopEvent.is_set():
                    break
                #ai_answer=get_completion(recognized_text)
                process_multiple_emotional_agents2(recognized_text,recognized_emotion)
            except Exception as e:
                print(e)
                #recognized_text=''

                
if __name__ == '__main__':
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.start()
    process_video()
    audio_thread.join()
