from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class DetectedObject:
    """
    Represents a single object detected in an image

    Attributes:
    label Name of the detected obj
    confidence: Detection confidence score
    bounding_box: Bounding Box in XYXY format
    mask Bonary segmentation mask for the object
    """

    label:str
    confidence: float
    bounding_box: torch.Tensor
    mask: np.ndarray