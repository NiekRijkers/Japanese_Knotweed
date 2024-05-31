from mmdet.apis import DetInferencer
import os

# Initialize the DetInferencer
# inferencer = DetInferencer(model='faster_rcnn_config.py', weights='work_dirs/faster_rcnn_config/epoch_12.pth')

# Perform inference

# Single Image 
# inferencer('data/data_japanese_knotweed/test/DJI_20231117104115_0006.JPG', out_dir='faster_rcnn_results/', no_save_vis=False)\

# Entire folder
# inferencer('data/data_japanese_knotweed/val_subset/', out_dir='faster_rcnn_inference/', no_save_vis=False)

def inference_folder(model, weights, input_dir, output_dir):
    # Initialize inferencer
    inferencer = DetInferencer(model=model, weights=weights)

    # Get all images paths from input directory
    image_paths = []
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image_paths.append(image_path)
        print(image_name)
    print(image_paths)
    image_paths.remove('data/data_japanese_knotweed/test/.ipynb_checkpoints')
    # Perform inference on all images from the directory
    inferencer(image_paths, out_dir = output_dir, no_save_vis = False)

model = 'configs/co_detr/co_detr_config.py'
weights = 'work_dirs/co_detr_config/new_small_isize.pth'
input_dir = 'data/data_japanese_knotweed/test/'
output_dir = 'co_detr_inference/'

inference_folder(model, weights, input_dir, output_dir) # Perform inference on entire folder

