import os
import json
import re


# Allow imports from src/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Change this if you want to query a different semantic scene
JSON_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "indoor_semantic_25",
    "semantic_frame_000016_png.rf.e55e7742936d294710c42737740cd419.json"
)


def load_scene(json_path):
    """Load a semantic scene from JSON."""
    with open(json_path, "r") as file:
        return json.load(file)


def find_objects(scene, label):
    """Find all detected objects matching a label."""
    label = label.lower().strip()

    return [
        obj for obj in scene["objects"]
        if obj["label"].lower() == label
    ]


def get_location(obj, image_width):
    """Return approximate horizontal object location."""

    x_min, y_min, x_max, y_max = obj["bounding_box"]

    center_x = (x_min + x_max) / 2

    if center_x < image_width / 3:
        return "left"

    elif center_x < (2 * image_width) / 3:
        return "center"

    else:
        return "right"


def answer_question(scene, question):
    """Interpret and answer simple semantic scene questions."""

    question = question.lower().strip()

    # What objects are in the scene?
    if question.startswith("what objects"):
        labels = scene["labels"]

        if not labels:
            return "No objects were detected."

        return "Objects detected: " + ", ".join(labels)

    # Where is/are the object?
    match = re.match(r"where (?:is|are) (?:the )?(.+?)[?]?$", question)

    if match:
        label = match.group(1).strip()
        objects = find_objects(scene, label)

        if not objects:
            return f"No {label} was detected."

        answers = []

        for index, obj in enumerate(objects, start=1):
            location = get_location(
                obj,
                scene["image_width"]
            )

            answers.append(
                f"{label} {index}: {location} side "
                f"(confidence: {obj['confidence']:.2f})"
            )

        return "\n".join(answers)

    # How many objects?
    match = re.match(
        r"how many (.+?)(?: are there| are in the scene)?[?]?$",
        question
    )

    if match:
        label = match.group(1).strip()

        # Basic plural handling
        plural_map = {
        "people": "person",
        "chairs": "chair",
        "tables": "table",
        "benches": "bench",
        "windows": "window",
        "doors": "door",
        "laptops": "laptop",
        "backpacks": "backpack",
        "bottles": "bottle"
    }

    label = plural_map.get(label, label)

    if label.endswith("s"):
        label = label[:-1]

    count = len(find_objects(scene, label))

    return f"Detected {count} {label}(s)."

    # Is there an object?
    match = re.match(
        r"is there (?:a |an |the )?(.+?)[?]?$",
        question
    )

    if match:
        label = match.group(1).strip()
        objects = find_objects(scene, label)

        if objects:
            return f"Yes. {len(objects)} {label}(s) detected."

        return f"No {label} was detected."

    return (
        "I don't understand that query yet. Try:\n"
        "- Where is the bench?\n"
        "- How many people are there?\n"
        "- Is there a laptop?\n"
        "- What objects are in the scene?"
    )


def main():

    print("=== SEMANTIC SCENE QUERY ===")

    scene = load_scene(JSON_PATH)

    print(f"\nLoaded scene: {os.path.basename(JSON_PATH)}")
    print(f"Objects detected: {scene['object_count']}")

    print(
        "\nYou can ask:\n"
        "- Where is the bench?\n"
        "- How many people are there?\n"
        "- Is there a laptop?\n"
        "- What objects are in the scene?\n"
        "- Type 'exit' to quit."
    )

    while True:

        question = input("\nAsk a question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting scene query.")
            break

        answer = answer_question(scene, question)

        print("\n" + answer)


if __name__ == "__main__":
    main()