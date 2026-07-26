class SceneQuery:
    """
    Provides simple query operations over a semantic scene
    """

    def __init__(self, scene):
        self.scene= scene

    def find_objects(self, label):
        """
        Find all objects in the sceen matching the label
        """
        return self.scene.get_objects_by_label(label.strip())

    def count_objects(self, label):
        """
        Count objects matching a label
        """
        return len(self.find_objects(label))

    def object_exists(self, label):
        """
        check whether an object exists in the scene.
        """
        return self.count_objects(label)>0

    def list_objects(self):
        """
        Return all unique object labels in the scene.
        """
        return sorted(set(obj.label for obj in self.scene.objects))
    
    def get_horizontal_location(self, obj):
        """
        Determine whether an object is on the left,
        center, or right side of the image.
        """

        #Bounding box i sstored in XYXY format
        x_min, y_min, x_max, y_max= obj.bounding_box.tolist()
        center_x=(x_min +x_max)/2

        #Scene image shape: height, wodth, channels
        image_width= self.scene.image.shape[1]

        if center_x <image_width /3:
            return "left"

        elif center_x <(2*image_width)/3:
            return "center"

        else:
            return "right"

    def locate_objects(self, label):
        """
        FInd objects with the requested label and
        return their approximate horizontal locations.
        """

        objects = self.find_objects(label)
        results=[]

        for obj in objects:
            results.append({
                "label": obj.label,
                "confidence": obj.confidence,
                "location": self.get_horizontal_location(obj)
            })

        return results
