import json
import os

# Define dataset path
data_dir = "image_dataset/train"

# Generate class names from the training dataset
if os.path.exists(data_dir):
    class_names = sorted(os.listdir(data_dir))
    class_names_dict = {str(i): name for i, name in enumerate(class_names)}

    # Save to JSON file
    with open("class_names.json", "w") as f:
        json.dump(class_names_dict, f)

    print("class_names.json regenerated successfully!")
else:
    print("Error: Training dataset path not found!")
