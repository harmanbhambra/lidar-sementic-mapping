import cv2
import numpy as np

class Visualizer:
    """
    Handles visualization of object detection and segmentation results
    """
    def __init__(self, alpha: float=0.5):
        """ Parameters: 
        alpha: float - Transparency of segmentation masks"""
        self.alpha=alpha

    def _generate_color(self, label: str):
        """
        Generate a deterministic RGB color for the given label
        """

        #Convert label into numeric hash value. Like chair is some integer
        #Person is a diff integer
        seed= abs(hash(label)) % (2**32)

        #Initialize numoys random number generator using hash value as a seed
        rng=np.random.default_rng(seed)
        #Generate three random integers between 0 and 255.
        #These represent the red green and blue colour channels
        color= tuple(int(c) for c in rng.integers(0,256, size=3))
        return color

    def _draw_masks(self, image, objects):
        """ 
        Draw segmentation masks with transparency
        """
        #output= image.copy() #Create a copy for the image 
        #iterate through every detected object.
        for obj in objects:
            label=obj.label
            mask=obj.mask
            #Generate a consistent color for the current object label
            color=self._generate_color(label)

            #Create an empty image that will contain only the coloured segmentation mask
            overlay = np.zeros_like(image)
            #Convert mask to boolean values. True represents pixels belonging to the detected object
            binary_mask = mask.astype(bool)
            #Color only the pixels blonging to the object
            overlay[binary_mask]=color
            #Blend colored mask with the original image. aplha controls the transparency
            image= cv2.addWeighted(image,1.0, overlay, self.alpha,0)

        return image

    def _draw_boxes(self, image,objects):
        """
        Draw bounding boxes

        Parameters:
        image: np.ndarray

        boxes: list [x1,y1,x2,y2]

        labels:list
        """
        #Create a copy of the original image
        #output=image.copy()

        #Iterate through every detected bouding box
        for obj in objects:

            box=obj.bounding_box
            label=obj.label

            #Generate a consistent color
            color= self._generate_color(label)

            #Convert the bounding box coordinated to integers
            #because OpenCv's drawing functions require integer pixel values
            x1,y1,x2,y2= box.int().tolist()

            #Draw the bounding box
            cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness=2)

        return image

    def _draw_labels(self, image, objects):
        """
        Draw labels and confidence scores
        """
        #Iterate through every  detected object
        for obj in objects:

            box=obj.bounding_box
            label=obj.label
            score=obj.confidence

            #Generate the same colour for objects mask and box
            color = self._generate_color(label)

            #Generate bounding box coordinates to integer pixel values
            x1,y1, _, _=box.int().tolist()

            #Create the label text like person(0.92)
            text=f"{label}: {score:.2f}"

            #Draw the label slightly above the top left corner of the box
            cv2.putText(image, text, (x1, max(y1-10,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2,cv2.LINE_AA)
    
        return image


    def draw(self, image, objects):
        """
        Draw masks, boxes, labels on an image
        Returns Annotated image np.ndarray
        """
        #Create a copy of the original image

        output=image.copy()
        #Draw segmentation masks
        output=self._draw_masks(output, objects)

        #Draw bounding boxes
        output=self._draw_boxes(output,objects)

        #Draw object labels and confidece scores
        output= self._draw_labels(output, objects)

        #Return the fully annotated image
        return output

    