from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import onnxruntime as ort

#def get_model_path(model_name):
#    return '../../models/affectnet_emotions/onnx/'+model_name+'.onnx'

   
    

    
class HSEmotionRecognizer:
    #supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b0_8_best_vgaf',providers=['CPUExecutionProvider']):
        self.is_mtl='_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
        else:
            self.idx_to_class={0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}

        if 'mbf_' in model_name:
            self.mean=[0.5, 0.5, 0.5]
            self.std=[0.5, 0.5, 0.5]
            self.img_size=112
        else:
            self.mean=[0.485, 0.456, 0.406]
            self.std=[0.229, 0.224, 0.225]
            if '_b2_' in model_name:
                self.img_size=260
            elif 'ddamfnet' in model_name:
                self.img_size=112
            else:
                self.img_size=224
        
        self.ort_session = ort.InferenceSession(model_name,providers=providers)
    
    def preprocess(self,img):
        x=cv2.resize(img,(self.img_size,self.img_size))/255
        for i in range(3):
            x[..., i] = (x[..., i]-self.mean[i])/self.std[i]
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
        