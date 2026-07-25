from dataclasses import dataclass
from typing import List
import numpy as np
from .detected_object import DetectedObject

@dataclass
class Scene:
    """
    Represents a visual scene and the semantic objects
    detected within it
    
    Attributes:
    image: Original image associated with the scene
    objects: Objects detected and segmented in the scene"""

    image: np.ndarray
    objects: List[DetectedObject]

    def get_objects_by_label(self, label:str):
        """
        Return all detected objects whose label matches
        the requested label.
        """

        #Create an empty list to store objects that match the label
        matching_objects=[]

        #Go through every detected object in the scene.
        for obj in self.objects:

            #Compare labels while ignoring uppercase/lowercase differences
            if obj.label.lower() == label.lower():
                matching_objects.append(obj)
        
        #Return all objects that matched the requested label.
        return matching_objects
    
    def count_objects(self):
        """
        Return the total number of detetcted objects in the scene
        """

        #self.objects contains every DetectedObject in the scene
        #len() therefore gives us the total number of detected objects.
        return len(self.objects)
    
    def get_labels(self):
        """
        Return the unique object labels present in the scene.
        """
        #Create an empty set
        #A set automatically prevents duplicate values.
        labels= set()

        #Go through every detected object.
        for obj in self.objects:
            labels.add(obj.label)
        return list(labels)
    def to_dict(self):
        """
        Convert the semantic scene into a dictionary
        that can later be saved as JSON
        """

        #Store the serialized represenatation of every object
        objects_data=[]

        #Go through every DetectedObject in the scene.
        for obj in self.objects:
            #Convert the pyTorch bounding box sensor into
            #normal Python list because JSON cannot store tensors.
            bounding_box=obj.bounding_box.tolist()

            #Create a dictionary describing the object
            object_data={ "label": obj.label, "confidence": obj.confidence,"bounding_box": bounding_box}

            objects_data.append(object_data)

        #Create the complete semantic representation of thescene
        scene_data={"object_count": self.count_objects(), "labels": self.get_labels(), "objects": objects_data}

        return scene_data


