from segment_anything  import sam_model_registry, SamPredictor
import cv2
import numpy as np

class SAMSegmenter:
    """Segments detetced objects using Segment Anything."""

    def __init__(self, checkpoint_path, model_type='vit_h', device='cpu'):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        

        print("SAM model loaded successfully.")

    def segment(self, image, boxes):
        """Generate segmentation masks for all detected boxes"""
        image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image_rgb)
        masks=[]

        for box in boxes:
            box=box.cpu().numpy()
            mask, score, _=self.predictor.predict(box=box, multimask_output=False)
            masks.append(mask[0])
        
        return masks