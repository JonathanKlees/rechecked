import os
import json
import pandas as pd
import numpy as np
from torchvision.ops import nms, box_iou, box_area
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


# this script is used to evaluate the benchmark for label error detection
# it loads the predicted boxes, the original ground truth and validated ground truth and determines the number of identified label errors and other metrics

def prepare_label_error_proposals(df, threshold_label_error, orig_iou_col = "iou_with_original_gt", score_col = "score"):
    # this function filters the predicted bounding boxes that have a IoU < thresh with the original GT and may thereby indicate label errors
    # the label error proposals are ordered in descending order according to the predicted score / confidence

    df = df[ df[orig_iou_col] < threshold_label_error] # those predicted boxes that have a true IoU below the threshold are considered label error proposals
    
    df = df.sort_values(by = score_col, ascending=False) # sort them according to the predicted score / confidence

    return df

def intersection_area(boxes1, boxes2):
    # boxes1: (N,4), boxes2: (M,4)
    # xyxy format
    N, M = boxes1.size(0), boxes2.size(0)

    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    return inter

def compare_orig_and_val_gt(validated_gt, original_gt, dont_care_regions):
    """
    This function calculates the IoU of each box in the validated GT with the original ground truth and the intersection over Area with don't care regions.
    It adds two new columns to the DataFrame: 'iou_with_original_gt' and 'ioa_with_dont_care' (intersection over area).
    """
    # get IoU with original GT for each predicted box to select label error proposals (IoU < 0.5)
    orig_ious = []
    for i, row in validated_gt.iterrows():
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

    validated_gt['iou_with_original_gt'] = orig_ious

    # get Intersection over Area of Prediction with dont care regions of the original GT for each predicted box to determine whether the object has been flagged as not of interest.
    dc_ioas = []
    for i, row in validated_gt.iterrows():
        # Get the corresponding predicted box
        pred_box = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
        
        # Get the ground truth boxes for the same image
        dc_boxes = dont_care_regions[dont_care_regions['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        if len(dc_boxes) > 0:
            area = box_area(pred_box)
            dc_boxes_tensor = torch.tensor(dc_boxes, dtype=torch.float32)
            inter = intersection_area(pred_box, dc_boxes_tensor).squeeze().numpy()
            dc_ioas.append( (inter.max() / area).item() ) # intersection with don't care over area of prediction
        else:
            dc_ioas.append(0)

    validated_gt['ioa_with_dont_care'] = dc_ioas

    return validated_gt

def get_ious_with_orig_and_val_gt(df, original_gt, validated_gt, dont_care_regions):
    """
    This function calculates the IoU of each predicted box in the DataFrame with the original ground truth (GT) and the validated GT.
    It also computes the Intersection over Area with Don't Care Regions.
    It adds three new columns to the DataFrame: 'iou_with_original_gt', 'iou_with_val_gt', and 'ioa_with_dont_care'.
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
    dc_ioas = []
    for i, row in df.iterrows():
        # Get the corresponding predicted box
        pred_box = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
        
        # Get the ground truth boxes for the same image
        dc_boxes = dont_care_regions[dont_care_regions['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        
        if len(dc_boxes) > 0:
            area = box_area(pred_box)
            dc_boxes_tensor = torch.tensor(dc_boxes, dtype=torch.float32)
            inter = intersection_area(pred_box, dc_boxes_tensor).squeeze().numpy()
            dc_ioas.append( (inter.max() / area).item() ) # intersection with don't care over area of prediction
        else:
            dc_ioas.append(0)

    df['ioa_with_dont_care'] = dc_ioas

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

def evaluate_proposed_label_errors(proposal_df, validated_gt):

    proposal_df["TP"] = [False]*len(proposal_df)
    overlooked_objects = 0
    misfitting_boxes = 0
    label_errors = 0
    validated_gt["matched"] = validated_gt["iou_with_original_gt"] > iou_threshold_misfitting_box # initialize boolean column with matches of original annotations

    # loop over label error proposals and match them to val GT to simulate refinement of annotations

    for i, row in proposal_df.iterrows(): # assuming it is ordered according to a meta model score

        proposal = torch.tensor([row['xmin'], row['ymin'], row['xmax'], row['ymax']], dtype=torch.float32).unsqueeze(0)
            
        # Get the val ground truth boxes for the same image
        val_gt_boxes = validated_gt[validated_gt['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].values
        indices = validated_gt[validated_gt['filename'] == row['filename']][['xmin', 'ymin', 'xmax', 'ymax']].index
        
        if len(val_gt_boxes) > 0: # only if there are val GT annoations in this image, consider them for IoU matching, else continue with the next proposal
            val_gt_boxes_tensor = torch.tensor(val_gt_boxes, dtype=torch.float32)
            iou = box_iou(proposal, val_gt_boxes_tensor).squeeze().numpy()

            if iou.max() > iou_threshold_TP: # if there is a matching box
                index = indices[iou.argmax()]
                if not validated_gt.at[index, "matched"]:  # Check if the box is not already matched
                    validated_gt.at[index, "matched"] = True  # Flag this box as matched
                    label_errors += 1
                    proposal_df.at[i, "TP"] = True

                    # here we would add IoU computation with orig. GT to distinguish overlooked and misfitting
    return proposal_df  

def filter_data_for_conditions(df):
    if filter_dont_care:
        df = df[df["ioa_with_dont_care"] < ioa_thresh_dont_care] # intersection over area: at least half of the box is outside
    if min_bbox_height:
        df = df[df["height"] >= min_bbox_height]
    return df

def perform_nms_on_dataframe(df, threshold=0.5, score_column = "score"):
    """
    Perform Non-Maximum Suppression on a DataFrame containing bounding boxes.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['xmin', 'ymin', 'xmax', 'ymax', 'score'].
        threshold (float): IoU threshold for suppression.
        
    Returns:
        pd.DataFrame: DataFrame after applying NMS.
    """
    if threshold == None:
        print("No NMS applied, threshold is None.")
        return df
    else:
        kept_rows = []

        for image_id, group in df.groupby(['filename']):
            boxes = torch.tensor(group[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
            scores = torch.tensor(group[score_column].values, dtype=torch.float32)
            
            if len(boxes) == 0:
                continue
            
            keep = nms(boxes, scores, threshold)
            kept_rows.append(group.iloc[keep.numpy()])

        return pd.concat(kept_rows).reset_index(drop=True)

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
    iou_threshold_TP = 0.1  # IoU threshold for true positive of label error proposals
    iou_threshold_misfitting_box = 0.5  # IoU threshold for misfitting boxes (FP of predictions w.r.t. original GT)
    ioa_thresh_dont_care = 0.5  # Intersection over Area of Bbox threshold for considering the intersection with dont care regions (inside / outside)
    nms_threshold = 0.5  # threshold for NMS filtering out similar predictions, if None, no NMS is applied
    
    # variable parameters for evaluation -> Changes data under consideration e.g. only larger objects or only objects with a certain probability in soft label annotation
    validated_gt_prob_threshold = 0.5  # Probability threshold for ground truth class pedestrian (0.5 and 0.8 available)
    min_bbox_height = 0 # minimal height of a bounding box to be considered as a pedestrian (according to KITTI Benchmark moderate and hard version). Filters all boxes.
    filter_dont_care = True

    # paths to the files
    path_to_predictions = "data/predictions/Cascade R-CNN.csv" # Method to benchmark

    print("Benchmarking "+ path_to_predictions.split("/")[-1].split(".csv")[0] + "... (change method in l. 210)" )

    path_to_original_gt = "data/original_gt.csv"
    path_to_validated_gt = "data/validated_gt.csv"
    path_to_split = "data/train_val_split.json"  # normally not needed but for safety, we filter the predictions by the val. split if the user provided predictions for all images

    ############ Start evaluation ############

    # 0. Load Data
    # 0.1 load the predictions and make sure that they are in the correct format and only for the validation split
    predictions_df = pd.read_csv(path_to_predictions)
    # make sure that predictions has the columns ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
    if not all(col in predictions_df.columns for col in ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'score']):
        raise ValueError("Predictions DataFrame must contain 'filename', 'xmin', 'ymin', 'xmax', 'ymax', and 'score' columns.")
    # next make sure that the predictions are only for the images in the validation split
    with open(path_to_split, "r") as f:
        split = json.load(f)
    image_ids = [f"{image_id}.png" for image_id in split["val_imgs"]]
    predictions_df = predictions_df[predictions_df['filename'].isin(image_ids)]
    predictions_df.reset_index(drop=True, inplace=True)
    
    # 0.2 load the original ground truth and the dont care regions
    original_gt = pd.read_csv(path_to_original_gt)
    dont_care_regions = pd.read_csv("data/original_gt_dont_care_regions.csv")
    
    # 0.3 load the validated ground truth
    validated_gt = pd.read_csv(path_to_validated_gt)
    validated_gt = compare_orig_and_val_gt(validated_gt, original_gt, dont_care_regions) # compute IoU and IoA for val. GT Boxes

    # 1. apply NMS to predictions_df to remove overlapping boxes and avoid duplicates
    predictions_df = perform_nms_on_dataframe(predictions_df, threshold=nms_threshold, score_column="score") 

    # 2. filter validated_gt for the conditions set by the parameters such as dont care regions, small boxes, as well as the probability threshold for the validated GT
    validated_gt = validated_gt[validated_gt["probability"] >= validated_gt_prob_threshold]
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering
    validated_gt = filter_data_for_conditions(validated_gt) # filters out small boxes and dont care regions in validated GT
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    # original_gt = filter_data_for_conditions(original_gt) # we are not filtering the original GT, leaving it unchanged. (there are a few small boxes and a few boxes in dont care regions though)

    # 3. Determine the number of label errors in the original GT based on the configuration of the validated GT.

    validated_gt = determine_label_errors_in_original_gt(original_gt, validated_gt, iou_threshold_misfitting_box)
    number_of_label_errors = len(validated_gt.query('label_error == True'))

    # 4. With the filtered validated_gt, we can now proceed with the evaluation of label error proposals. First, we filter the predictions_df

    # 4.1 get IoUs with original and validated GT and dont care regions 
    predictions_df = get_ious_with_orig_and_val_gt(predictions_df, original_gt, validated_gt, dont_care_regions) 

    # 4.2 filter predictions_df for the conditions set by the parameters such as dont care regions, small boxes
    predictions_df = filter_data_for_conditions(predictions_df)
    predictions_df.reset_index(drop=True, inplace=True)  # reset index after filtering


    # 4.3 sort predictions by score and keep only those with IoU < 0.5 (iou_threshold_misfitting_box) with the original GT as label error proposals (the others match)
    predictions_df = prepare_label_error_proposals(predictions_df, iou_threshold_misfitting_box)
    predictions_df.reset_index(drop=True, inplace=True)  # reset index after filtering

    
    # 5. evaluate the label error proposals

    # 5.1 Apply IoU matching with the validated GT to evaluate the label error proposals
    predictions_df = evaluate_proposed_label_errors(predictions_df, validated_gt)

    # 5.2 Compute scores
    print(f"Number of label error proposals: {len(predictions_df)}")
    print(f"Number of identified label errors: {len(predictions_df.query('TP == True'))}")
    print(f"Total number of label errors: {number_of_label_errors}", "(Overlooked: ", len(validated_gt.query('label_error_type == "overlooked pedestrian"')), ", Misfitting: ", len(validated_gt.query('label_error_type == "misfitting box"')), ")")

    precision = len(predictions_df.query('TP == True')) / len(predictions_df) if len(predictions_df) > 0 else 0
    recall = len(predictions_df.query('TP == True')) / number_of_label_errors if number_of_label_errors > 0 else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # AUROC and AUPRC
    y = predictions_df["TP"].astype(int).values
    y_hat = predictions_df["score"].values
    auroc = roc_auc_score(y, y_hat)
    auprc = average_precision_score(y, y_hat)
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
