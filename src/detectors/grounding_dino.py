#This file's only responsibility is detections. We dont use boxes here like we did in the test_groundingDINO file.
#It just finds objects
from groundingdino.util.inference import(load_model, load_image, predict,)
import os
from torchvision.ops import box_convert
import torch

class GroundingDINODetector:
 
 def __init__(self, config_path, weights_path):
  #Load the GroundingDINO model once.
  self.model=load_model(config_path, weights_path)
  print("GroundingDINO model loaded successfully")

 def detect(self, image_path, text_prompt, box_threshold=0.35,text_threshold=0.25,):
  #Detect objects in an image.
  image_source, image= load_image(image_path)
  boxes,logits, phrases=predict(model= self.model, image=image, caption=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold,)
  h,w, _= image_source.shape
  boxes=boxes*torch.tensor([w,h,w,h])
  boxes=box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt='xyxy')

  return image_source, boxes, logits, phrases