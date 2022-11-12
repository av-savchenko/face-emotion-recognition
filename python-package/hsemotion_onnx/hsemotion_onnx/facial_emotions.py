from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import urllib

#def get_model_path(model_name):
#    return '../../models/affectnet_emotions/onnx/'+model_name+'.onnx'

def get_model_path(model_name):
    model_file=model_name+'.onnx'
    cache_dir = os.path.join(os.path.expanduser('~'), '.hsemotion')
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        url='https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/'+model_file+'?raw=true'
        print('Downloading',model_name,'from',url)
        urllib.request.urlretrieve(url, fpath)
    return fpath        
    

    
class HSEmotionRecognizer:
    #supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b0_8_best_vgaf'):
        self.is_mtl='_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
        else:
            self.idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

        self.img_size=224 if '_b0_' in model_name else 260
        
        path=get_model_path(model_name)
        self.ort_session = ort.InferenceSession(path,providers=['CPUExecutionProvider'])
    
    def preprocess(self,img):
        x=cv2.resize(img,(self.img_size,self.img_size))/255
        x[..., 0] = (x[..., 0]-0.485)/0.229
        x[..., 1] = (x[..., 1]-0.456)/0.224
        x[..., 2] = (x[..., 2]-0.406)/0.225
        return x.transpose(2, 0, 1).astype("float32")[np.newaxis,...]

    def predict_emotions(self,face_img, logits=True):
        scores=self.ort_session.run(None,{"input": self.preprocess(face_img)})[0][0]
        if self.is_mtl:
            x=scores[:-2]
        else:
            x=scores
        pred=np.argmax(x)
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2]=e_x
            else:
                scores=e_x
        return self.idx_to_class[pred],scores
                
    def predict_multi_emotions(self,face_img_list, logits=True):
        imgs = np.concatenate([self.preprocess(face_img) for face_img in face_img_list],axis=0)
        scores=self.ort_session.run(None,{"input": imgs})[0]
        if self.is_mtl:
            preds=np.argmax(scores[:,:-2],axis=1)
        else:
            preds=np.argmax(scores,axis=1)
        if self.is_mtl:
            x=scores[:,:-2]
        else:
            x=scores
        pred=np.argmax(x[0])
        
        if not logits:
            e_x = np.exp(x - np.max(x,axis=1)[:,np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:,None]
            if self.is_mtl:
                scores[:,:-2]=e_x
            else:
                scores=e_x

        return [self.idx_to_class[pred] for pred in preds],scores
        