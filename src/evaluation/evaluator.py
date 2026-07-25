import os
import json
from collections import Counter

class SceneEvaluator:
    """
    Calculates statistics from semantic scene JSON files.
    """

    def __init__(self, results_folder):
        """
        Parameters:
        results_folder: Folder contains semantic JSON files.
        """
        self.results_folder=results_folder

    def evaluate(self):
        """
        Read all semantic JSON files and calculate dataset level detection statistics"""

        #Find all semantic JSON files in the results folder
        json_files=[file 
                    for file in os.listdir(self.results_folder)
                    if file.startswith("semantic_") and file.endswith(".json")]
        
        #Store statistics across all scenes.
        total_detections=0
        label_counts= Counter()
        confidence_scores=[]

        #Process every semantic scene.
        for json_file in json_files:
            file_path=os.path.join(self.results_folder, json_file)

            #Load the semantic scene.
            with open(file_path, "r") as file:
                scene_data=json.load(file)

            #Examine every detected object in this scene.
            for obj in scene_data["objects"]:

                total_detections +=1
                #Count how often each label appears
                label_counts[obj["label"]] += 1

                #Store confidence so we can calculate the average confidence later
                confidence_scores.append(obj["confidence"])
            
            #Avoid division by zero if there are no detection
        average_confidence=(sum(confidence_scores)/len(confidence_scores)
                                if confidence_scores
                                else 0
                                )
            
        average_objects_per_image=(total_detections/len(json_files)
                                       if json_files
                                       else 0)
            
        return { "images_processed": len(json_files),
                    "total_detections": total_detections,
                    "average_objects_per_image": average_objects_per_image,
                     "average_confidence": average_confidence,
                    "label_counts": dict(label_counts)}
            