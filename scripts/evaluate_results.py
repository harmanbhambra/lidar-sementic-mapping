import os
import sys

# Get project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Add src to Python's import path
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from evaluation import SceneEvaluator


# Experiment that we want to evaluate
RESULTS_FOLDER = os.path.join(
    PROJECT_ROOT,
    "results",
    "indoor_semantic_25"
)


evaluator = SceneEvaluator(
    results_folder=RESULTS_FOLDER
)

results = evaluator.evaluate()
print("DEBUG results:", results)
print("\n=== EXPERIMENT SUMMARY ===")

print("\n=== EXPERIMENT SUMMARY ===")

print(f"Images processed: {results['images_processed']}")
print(f"Total detections: {results['total_detections']}")

print(
    f"Average objects per image: "
    f"{results['average_objects_per_image']:.2f}"
)

print(
    f"Average confidence: "
    f"{results['average_confidence']:.2f}"
)

print("\nLabel counts:")

for label, count in results["label_counts"].items():
    print(f"  {label}: {count}")