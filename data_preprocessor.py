import json
import os
from PIL import Image
import shutil
import random

def convert_to_coco_format(coords) -> list:
        """This function transforms a bounding box with four xy coordinates to COCO format.

        Args:
            coords (list): A list containing xy-coordinates of a bbox clockwise (starting from left-top).
                Format: [{'X': int, 'Y': int}, {'X': int, 'Y': int}, {'X': int, 'Y': int}, {'X': int, 'Y': int}]

        Returns:
            list: A list containing the bbox in COCO format: [x_min, y_min, w, h]
        """
        x_coordinates = [coord["X"] for coord in coords]
        y_coordinates = [coord["Y"] for coord in coords]

        x_min = min(x_coordinates)
        y_min = min(y_coordinates)
        x_max = max(x_coordinates)
        y_max = max(y_coordinates)

        width = x_max - x_min
        height = y_max - y_min

        return [x_min, y_min, width, height]

def convert_single_ann_file(ann_file, coco_data, img_path): 
    
    # Open the original annotation file
    f = open(
        ann_file,
    )
    old_annotations = json.load(f)

    # Take only the images
    image_extensions = [".png", ".jpg"]
    image_files = [
        file for file in os.listdir(img_path) if os.path.splitext(file)[-1].lower() in image_extensions
    ]

    for img in image_files:
        img_id = len(coco_data["images"])
        image_path = os.path.join(img_path, img)
        image = Image.open(image_path)
        width, height = image.size

        # Add image information to COCO annotations
        coco_data["images"].append({"id": img_id, "file_name": img, "height": height, "width": width})

        for image_info in old_annotations:
            if image_info["OriginalFileName"] == img:
                for old_ann in image_info["AnnotationData"]:
                    label = old_ann["Label"]
                    coordinates = old_ann["Coordinates"]
                    coco_bbox = convert_to_coco_format(coordinates)

                    # Add category information to COCO annotations
                    if label not in [cat["name"] for cat in coco_data["categories"]]:
                        category_id = len(coco_data["categories"])
                        coco_data["categories"].append(
                            {
                                "id": category_id,
                                "name": str(label),
                            }
                        )

                    # Add annotation information to COCO annotations
                    cat_id = next(d["id"] for d in coco_data["categories"] if d["name"] == label)
                    coco_data["annotations"].append(
                        {
                            "id": len(coco_data["annotations"]),
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": [int(point) for point in coco_bbox],
                            "area": int(coco_bbox[2] * coco_bbox[3]),
                            "iscrowd": 0,
                        }
                    )
    return coco_data

def convert_anns_scenius_to_coco(ann_files, coco_ann_file, img_path):  # noqa: ANN201
    """This function converts a json annotation file retrieved from Scenius
    and converts it to a COCO format annotation file.

    Args:
        ann_file (str): The path to the json annotation file needs to be converted.
        coco_ann_file (str): The path to store the COCO annotation file.
        img_path (str): The folder containing the images related to the annotation file.
    """
    # Initialize COCO annotations
    coco_data = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}

    if isinstance(ann_files, list):
        for ann_file in ann_files:
            coco_data = convert_single_ann_file(ann_file, coco_data, img_path)
            print(len(coco_data['annotations']))
    else:
        coco_data = convert_single_ann_file(ann_files, coco_data, img_path)

    print(len(coco_data['annotations']))
    # Write COCO annotations to a JSON file
    with open(coco_ann_file, "w") as output_file:
        json.dump(coco_data, output_file)

def remove_duplicate_annotations(ann_file):
    """This function removes duplicate images from the annotation file
    such that there are no mulitple annotations for the same image.
    
    Args:
        ann_file(str): The path to the json annotation file from which duplicates need to be removed."""
    # Open annotations file
    with open(ann_file, 'r') as file:
        annotation_file = json.load(file)
    
    # Create a dictionary to store unique entries based on OriginalFileName
    unique_images = {}
    removed_images = []
    for image in annotation_file:
            original_file_name = image["OriginalFileName"]
            if original_file_name not in unique_images:
                 unique_images[original_file_name] = image
            else:
                 removed_images.append(original_file_name)
    
    # Convert the dictionary back to a list
    unique_annotations = list(unique_images.values())    

    # Save the unique data to a new JSON file
    with open('unique_annotations.json', 'w') as file:
        json.dump(unique_annotations, file, indent=2)

    # Print old and new number of images and the removed images  
    print(f"Number of images before: {len(annotation_file)}")
    print(f"Number of images after removing duplicates: {len(unique_annotations)}")
    print(f"List of removed images: {removed_images}")



def train_val_split(images, train_path, val_path, test_path, train_percentage):             
    """ This function splits the images into training and validation data.
    It gets the images from the original folder and puts them in two different folders.
    
    Args: 
       images: The path to the folder with all the images
       train_pat: The path to the training folder
       val_path: The path to the validation folder
       train_percentage: The percentage of images in the training set"""
    


    # Get the list of all image files in the original folder
    image_files = [f for f in os.listdir(images)]

    # Calculate the number of images for training and for the validation/testing
    num_training_images = int(len(image_files) * train_percentage)
    num_validation_images = int(0.5*(len(image_files) - num_training_images)) # Multiply by 0.5, because half of the remaining images are for validation and half for testing

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Split the image files into training and validation sets
    training_set = image_files[:num_training_images]
    validation_set = image_files[num_training_images:num_training_images+num_validation_images]
    testing_set = image_files[num_training_images+num_validation_images:]

    # Copy images to the training folder
    for image_file in training_set:
        src_path = os.path.join(images, image_file)
        dst_path = os.path.join(train_path, image_file)
        shutil.copy(src_path, dst_path)

    # Copy images to the validation folder
    for image_file in validation_set:
        src_path = os.path.join(images, image_file)
        dst_path = os.path.join(val_path, image_file)
        shutil.copy(src_path, dst_path)

    # Copy images to the testing folder
    for image_file in testing_set:
        src_path = os.path.join(images, image_file)
        dst_path = os.path.join(test_path, image_file)
        shutil.copy(src_path, dst_path)

    print("Dataset split into training and validation sets.")


def new_train_val_split(images, n_segments, train_path, val_path, test_path, train_percentage):             
    """ This function splits the images into training and validation data.
    It gets the images from the original folder and puts them in two different folders.
    
    Args: 
       images: The path to the folder with all the images
       train_pat: The path to the training folder
       val_path: The path to the validation folder
       train_percentage: The percentage of images in the training set"""  
    

    # Get the list of all image files in the original folder
    total_image_files = [f for f in os.listdir(images)]
    total_n_images = len(total_image_files)
    segment_size = total_n_images // n_segments

    # Split data up into n number of segments
    segments = []
    for i in range(0, total_n_images, segment_size):
        segment = total_image_files[i:i+segment_size]
        segments.append(segment)

    # For every segment, take 70% for training, 15% for testing and 15% for validating
    for segment in segments:

        # Calculate the number of images for training and for the validation/testing
        num_training_images = int(len(segment) * train_percentage)
        num_validation_images = int(0.5*(len(segment) - num_training_images)) # Multiply by 0.5, because half of the remaining images are for validation and half for testing

        # Split the image files into training and validation sets
        training_set = segment[:num_training_images]
        validation_set = segment[num_training_images:num_training_images+num_validation_images]
        testing_set = segment[num_training_images+num_validation_images:]

        # Copy images to the training folder
        for image_file in training_set:
            src_path = os.path.join(images, image_file)
            dst_path = os.path.join(train_path, image_file)
            shutil.move(src_path, dst_path)

        # Copy images to the validation folder
        for image_file in validation_set:
            src_path = os.path.join(images, image_file)
            dst_path = os.path.join(val_path, image_file)
            shutil.move(src_path, dst_path)

        # Copy images to the testing folder
        for image_file in testing_set:
            src_path = os.path.join(images, image_file)
            dst_path = os.path.join(test_path, image_file)
            shutil.move(src_path, dst_path)

    print("Dataset split into training and validation sets.")

def create_subset_annotations(ann_file_path, subset_folder, file_name, output_folder):
    """This function creates an annotation file for a specified subset of the data using the annotation file from all images.
    
    Args:
        ann_file_path(str): The path to the json annotation file containing the annotations from all images.
        subset_folder(str): The path to the folder where the subset annotation file needs to go.
        file_name(str): The name of the output file."""
    # Open annotations file
    with open(ann_file_path, 'r') as file:
        annotation_file = json.load(file)
        
    # Get a list of the target images
    files = os.listdir(subset_folder)
    target_images = [file for file in files if file.lower().endswith('.jpg')]
    subset_images = {}
    # Create a dictionary to store unique entries based on OriginalFileName
    for image in annotation_file:
            original_file_name = image["OriginalFileName"]
            if original_file_name in target_images:
                 subset_images[original_file_name] = image
    
    # Convert the dictionary back to a list
    subset_annotations = list(subset_images.values())    

    # Save the unique data to a new JSON file
    file_path = os.path.join(output_folder, file_name)
    with open(file_path, 'w') as file:
        json.dump(subset_annotations, file, indent=2)

def create_train_subset(images, subset_size, output_folder):
    """ This function creates a subset of the training set, based on a random seed """    
    # Set random seed for reproducability
    random.seed(10)
    # Get files
    files = os.listdir(images)
    selected_files = random.sample(files, subset_size)
    
    # Copy selected files to the destination folder
    for file_name in selected_files:
        source_file = os.path.join(images, file_name)
        destination_file = os.path.join(output_folder, file_name)
        shutil.copyfile(source_file, destination_file)

create_train_subset("data/data_japanese_knotweed/train", 100, "data/data_japanese_knotweed/subset_train")
create_subset_annotations("data/data_japanese_knotweed/new_unique_annotations.json", "data/data_japanese_knotweed/subset_train", "subset_train_annotations.json", "data/data_japanese_knotweed")
convert_anns_scenius_to_coco("data/data_japanese_knotweed/subset_train_annotations.json", "data/data_japanese_knotweed/annotations/subset_train_annotations_coco.json", "data/data_japanese_knotweed/subset_train")

