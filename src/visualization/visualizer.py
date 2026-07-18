import cv2
from groundingdino.util.inference import annotate

class DetectionVisualizer:
    '''Visualizes and saves object detection results.'''

    def save_detection_result(self, image_source, boxes, logits, phrases, output_path):
        '''Annotate an image with the detected objects and save it'''

        annotated_frame= annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases,)

        cv2.imwrite(output_path, annotated_frame)