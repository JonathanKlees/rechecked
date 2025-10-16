import os
import json
import pandas as pd
from pathlib import Path
import torch
from torchvision.ops import box_iou

def load_kitti_annotations(kitti_path):
    label_path = Path(kitti_path) / "training/label_2"
    annotations = {}
    for label_file in label_path.glob("*.txt"):
        with open(label_file, "r") as f:
            annotations[label_file.stem] = f.readlines()
    return annotations

def filter_images(kitti_path, split_file):
    with open(split_file, "r") as f:
        split_data = json.load(f)
    val_images = set(split_data.get("val_imgs", []))
    return val_images

def process_annotations(annotations, val_images):
    pedestrian_data = []
    dont_care_data = []

    for image_id, lines in annotations.items():
        if image_id not in val_images:
            continue
        for line in lines:
            parts = line.strip().split()
            class_name = parts[0]
            bbox = parts[4:8]  # Extract bounding box coordinates
            if class_name == "Pedestrian":
                pedestrian_data.append([image_id + ".png"] + [float(coord) for coord in bbox] + ["Pedestrian", 2, float(bbox[3]) - float(bbox[1])]) # here, we also store the bbox height
            elif class_name == "DontCare":
                dont_care_data.append([image_id + ".png"] + [float(coord) for coord in bbox] + ["DontCare", -1])
    
    return pedestrian_data, dont_care_data

def save_to_csv(data, output_file, columns):
    df = pd.DataFrame(data, columns=columns)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

def get_ious_with_dont_care(original_gt, dont_care_regions):
    """
    This function calculates the IoU of each bounding box in the GT with the don't care regions.
    It a new column to the DataFrame: 'iou_with_dont_care'.
    """
    
    # get IoU with dont care regions for each GT box to determine whether the object has been flagged as not of interest.
    dc_ious = []
    for i, row in original_gt.iterrows():
        # Get the corresponding bounding box
        gt_box = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
        
        # Get the ground truth boxes for the same image
        dc_boxes = dont_care_regions[dont_care_regions['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        if len(dc_boxes) > 0:
            dc_boxes_tensor = torch.tensor(dc_boxes, dtype=torch.float32)
            iou = box_iou(gt_box, dc_boxes_tensor).squeeze().numpy()
            dc_ious.append(iou.max())
        else:
            dc_ious.append(0)

    original_gt['iou_with_dont_care'] = dc_ious

    return original_gt

def main():
    kitti_path = input("Enter the path to the KITTI dataset: ").strip()
    split_file = "data/train_val_split.json"

    if not os.path.exists(split_file):
        print(f"Split file not found: {split_file}")
        return

    annotations = load_kitti_annotations(kitti_path)
    print(len(annotations), "annotation files loaded.")
    val_images = filter_images(kitti_path, split_file)
    pedestrian_data, dont_care_data = process_annotations(annotations, val_images)

    save_to_csv(pedestrian_data, "data/original_gt.csv", ["filename", "xmin", "ymin", "xmax", "ymax", "class_name", "class", "height"])
    save_to_csv(dont_care_data, "data/original_gt_dont_care_regions.csv", ["filename","xmin", "ymin", "xmax", "ymax", "class_name", "class"])

    # lastly compute the IoU with don't care regions
    if dont_care_data:
        dont_care_df = pd.DataFrame(dont_care_data, columns=["filename","xmin", "ymin", "xmax", "ymax", "class_name", "class"])
        pedestrian_df = pd.DataFrame(pedestrian_data, columns=["filename", "xmin", "ymin", "xmax", "ymax", "class_name", "class", "height"])
        
        # compute IoU with don't care regions and store
        pedestrian_df = get_ious_with_dont_care(pedestrian_df, dont_care_df)
        pedestrian_df.to_csv("data/original_gt.csv", index=False)

    print("Processing of KITTI annotations completed. Annotations stored in data directory.")

if __name__ == "__main__":
    main()