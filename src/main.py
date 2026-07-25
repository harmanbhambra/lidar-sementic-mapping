import os
import cv2
import json
from detectors.grounding_dino import GroundingDINODetector
from visualization import Visualizer
from segmentation import SAMSegmenter
from semantic import DetectedObject , Scene

print("=== START OF MAIN ===")
#where the project is located
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

CONFIG_PATH=os.path.join(PROJECT_ROOT, "GroundingDINO","groundingdino", "config", "GroundingDINO_SwinT_OGC.py")


WEIGHTS_PATH=os.path.join(PROJECT_ROOT, "models", "groundingdino_swint_ogc.pth")

SAM_CHECKPOINT=os.path.join(PROJECT_ROOT, "models", "sam_vit_h_4b8939.pth")

DATASET_PATH= r"C:\Users\DELL\mu-hall-yolov8"

IMAGE_PATH=os.path.join(DATASET_PATH, "train","images")

#OUTPUT_FOLDER= os.path.join(PROJECT_ROOT, "results")
#os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#EXPERIMENT_NAME= "open_vocab_test"
#OUTPUT_FOLDER=os.path.join(PROJECT_ROOT, "results", EXPERIMENT_NAME)
#os.makedirs(OUTPUT_FOLDER, exist_ok=True)
EXPERIMENT_NAME2= "open_vocab_chair_bench"
OUTPUT_FOLDER=os.path.join(PROJECT_ROOT, "results", EXPERIMENT_NAME2 )
os.makedirs(OUTPUT_FOLDER,exist_ok=True )

image_files = sorted(
    file
    for file in os.listdir(IMAGE_PATH)
    if file.lower().endswith((".jpg", ".png", ".jpeg"))
)

print(f"Found {len(image_files)} images.")

#Here we are going to instantiate the detector. It calls the init method which loads the config, the trained weights and creates the model
#print("Before Detector")
detector= GroundingDINODetector(config_path= CONFIG_PATH, weights_path=WEIGHTS_PATH)
#print("After Detector")
#Create a segmenter
#print("Before SAM")
segmenter=SAMSegmenter(checkpoint_path=SAM_CHECKPOINT, model_type='vit_h', device='cpu')
#print("After Sam")
#Create a visualizer object

print("After visualizer")
visualizer = Visualizer(alpha=0.5)
print("After Visualiser")

#Start a small batch to test the pipeline safely.
test_images=image_files[:5]
print(f"Processing {len(test_images)} images:")

#Process each image one at a time
for index, image_file in enumerate(test_images, start=1):
    print(f"\n{'='* 50}")
    print(f"Processing image {index}/{len(test_images)}: {image_file}")
    print(f"{'=' *50}")

    #Create full path to the current image
    image_path=os.path.join(IMAGE_PATH, image_file)

   #Step 1: Open Vocabulary Object Detection
    #called the detector
    print("Step 1: Starting detection...")
    image_source, boxes, logits, phrases = detector.detect(
        image_path=image_path,
        #text_prompt="chair . table . person . laptop . door . window",
        #text_prompt=("backpack . bottle . bag . fire extinguisher ."
        #"notice board . bench . clock . television"),
        text_prompt=("chair . bench . person . seating"),
        box_threshold=0.35,
        text_threshold=0.25
    )
    print(f"Detection Complete. Found {len(boxes)} objects")
    # Step 2: Segmentation
    print("Step 2: Starting segmentation...")
    masks=segmenter.segment(image=image_source, boxes=boxes)
    print(f"generated {len(masks)} segmentation masks.")
   
   
    #Step 3: Create Semantic Objects
    detected_objects=[]
    # Combine the label, confidence, bounding box and mask
    # belonging to each detection into one DetectedObject.
    for box, score, label, mask in zip(boxes, logits, phrases, masks):

        detected_object = DetectedObject(
            label=label,
            confidence=score.item(),
            bounding_box=box,
            mask=mask
        )

        detected_objects.append(detected_object)

    #Step 4: Create the semantic Scene
    # Combine the input image and all detected objects
    # into a single semantic representation of the scene.
    scene = Scene(
        image=image_source,
        objects=detected_objects
    )

    print(f"\nTotal objects: {scene.count_objects()}")
    print(f"Labels in scene: {scene.get_labels()}")

    #Step 5: Visualize the Scene
    annotated_image = visualizer.draw(
            image=scene.image,
            objects=scene.objects
        )

        # Create a unique output path for this image.
    image_output_path = os.path.join(
            OUTPUT_FOLDER,
            f"detected_{image_file}"
        )

    cv2.imwrite(
            image_output_path,
            annotated_image
        )

    print(f"Annotated image saved to: {image_output_path}")


        
        # STEP 6: SAVE SEMANTIC SCENE AS JSON
    
    scene_data = scene.to_dict()

        # Remove the original extension so that we can
        # create a clean JSON filename.
    image_name = os.path.splitext(image_file)[0]

    json_output_path = os.path.join(
            OUTPUT_FOLDER,
            f"semantic_{image_name}.json"
        )

    with open(json_output_path, "w") as json_file:
            json.dump(
                scene_data,
                json_file,
                indent=4
            )

    print(f"Semantic scene saved to: {json_output_path}")
    
        # PRINT DETECTED OBJECTS
    print("Detected objects:")

    for obj in scene.objects:
         print(f"  - {obj.label}: {obj.confidence:.2f}")


print("\nBatch processing complete.")

