import os
import json
import pandas as pd
import numpy as np
from torchvision.ops import nms, box_iou
import torch


# this script compares original and validated ground truth annotations and determines the number of label errors in the original ground truth.
# It distinguishes overlooked pedestrians and misfitting boxes and considers different conditions such as dont care regions, small boxes, and the probability threshold for the validated ground truth.
# Also, if specified, it plots the identified label errors (a random sample of 12 and all for the most conservative configuration)

def get_ious_with_orig_and_val_gt(df, original_gt, validated_gt, dont_care_regions):
    """
    This function calculates the IoU of each predicted box in the DataFrame with the original ground truth (GT) and the validated GT.
    It adds two new columns to the DataFrame: 'iou_with_original_gt' and 'iou_with_val_gt'.
    """
    # get IoU with original GT for each predicted box to select label error proposals (IoU < 0.5)
    orig_ious = []
    for i, row in df.iterrows():
        # Get the corresponding predicted box
        pred_box = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
        
        # Get the ground truth boxes for the same image
        original_gt_boxes = original_gt[original_gt['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        if len(original_gt_boxes) > 0:
            original_gt_boxes_tensor = torch.tensor(original_gt_boxes, dtype=torch.float32)
            iou = box_iou(pred_box, original_gt_boxes_tensor).squeeze().numpy()
            orig_ious.append(iou.max())
        else:
            orig_ious.append(0)

    df['iou_with_original_gt'] = orig_ious

    # get IoU with dont care regions of the original GT for each predicted box to determine whether the object has been flagged as not of interest.
    dc_ious = []
    for i, row in df.iterrows():
        # Get the corresponding predicted box
        pred_box = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
        
        # Get the ground truth boxes for the same image
        dc_boxes = dont_care_regions[dont_care_regions['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        if len(dc_boxes) > 0:
            dc_boxes_tensor = torch.tensor(dc_boxes, dtype=torch.float32)
            iou = box_iou(pred_box, dc_boxes_tensor).squeeze().numpy()
            dc_ious.append(iou.max())
        else:
            dc_ious.append(0)

    df['iou_with_dont_care'] = dc_ious

    # get IoU with validated GT for each predicted box in the overlooked DataFrame to determine label errors of this type
    val_ious = []
    for i, row in df.iterrows():
        # Get the corresponding predicted box
        pred_box = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
        
        # Get the validated ground truth boxes for the same image
        val_gt_boxes = validated_gt[validated_gt['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        if len(val_gt_boxes) > 0:
            val_gt_boxes_tensor = torch.tensor(val_gt_boxes, dtype=torch.float32)
            iou = box_iou(pred_box, val_gt_boxes_tensor).squeeze().numpy()
            val_ious.append(iou.max())
        else:
            val_ious.append(0)

    df['iou_with_val_gt'] = val_ious

    return df

def filter_data_for_conditions(df):
    if filter_dont_care:
        df = df[df["iou_with_dont_care"] < iou_thresh_dont_care]
    if min_bbox_height:
        df = df[df["height"] >= min_bbox_height]
    return df

def determine_label_errors_in_original_gt(original_gt, validated_gt, iou_threshold_misfitting_box):
    bool_label_errors = [False] * len(validated_gt)
    label_error_type = ["no"] * len(validated_gt)
    label_errors = 0

    matched_original_indices = set()

    for i, row in validated_gt.iterrows():
        filename = row['filename']
        validated_box = torch.tensor(row[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float), dtype=torch.float32).unsqueeze(0)

        original_boxes = original_gt[original_gt['filename'] == filename]

        best_iou = 0
        best_match_idx = None

        for j, original_box in original_boxes.iterrows():
            if j in matched_original_indices:
                continue  # already matched

            original_box_tensor = torch.tensor([
                float(original_box['xmin']), float(original_box['ymin']),
                float(original_box['xmax']), float(original_box['ymax'])
            ], dtype=torch.float32).unsqueeze(0)

            iou = box_iou(validated_box, original_box_tensor)[0, 0].item()
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j

        if best_iou >= iou_threshold_misfitting_box:
            matched_original_indices.add(best_match_idx)  # mark match
        else:
            label_errors += 1
            bool_label_errors[i] = True
            if best_iou == 0:
                label_error_type[i] = "overlooked pedestrian"
            else:
                label_error_type[i] = "misfitting box"

    validated_gt['label_error'] = bool_label_errors
    validated_gt['label_error_type'] = label_error_type
    return validated_gt

if __name__ == "__main__":

    # fixed parameters for evaluation
    iou_threshold_misfitting_box = 0.5  # IoU threshold for misfitting boxes (FP of predictions w.r.t. original GT)
    iou_thresh_dont_care = 0.5  # IoU threshold for considering the intersection with dont care regions
    
    # variable parameters for evaluation -> Changes data under consideration e.g. only larger objects or only objects with a certain probability in soft label annotation
    validated_gt_prob_threshold = 0.8  # Probability threshold for ground truth class pedestrian (0.5 and 0.8 available)
    min_bbox_height = 25 # minimal height of a bounding box to be considered as a pedestrian (according to KITTI Benchmark moderate and hard version). Filters all boxes.
    filter_dont_care = False

    # paths to the files
    path_to_original_gt = "data/original_gt.csv"
    path_to_validated_gt = "data/validated_gt.csv"

    ############ Load original and validated annotations and filter the annotated annotations according to the parameters ############

    # load the original ground truth and the dont care regions
    original_gt = pd.read_csv(path_to_original_gt)
    dont_care_regions = pd.read_csv("data/original_gt_dont_care_regions.csv")
    
    # load the validated ground truth
    validated_gt = pd.read_csv(path_to_validated_gt)

    # filter validated_gt for the conditions set by the parameters such as dont care regions, small boxes, as well as the probability threshold for the validated GT
    validated_gt = validated_gt[validated_gt["probability"] >= validated_gt_prob_threshold]
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering
    validated_gt = filter_data_for_conditions(validated_gt) # filters out small boxes and dont care regions in validated GT
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    original_gt = filter_data_for_conditions(original_gt) # filters out small boxes and boxes in dont care regions in orig. GT
    original_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    #  Determine the number of label errors in the original GT based on the configuration of the validated GT.

    validated_gt = determine_label_errors_in_original_gt(original_gt, validated_gt, iou_threshold_misfitting_box)
    number_of_label_errors = len(validated_gt.query('label_error == True'))

    print(f"Total number of label errors: {number_of_label_errors}", "(Overlooked: ", len(validated_gt.query('label_error_type == "overlooked pedestrian"')), ", Misfitting: ", len(validated_gt.query('label_error_type == "misfitting box"')), ")")
