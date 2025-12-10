import os
import json
import pandas as pd
import numpy as np
from torchvision.ops import nms, box_iou, box_area
import torch
import cv2
import matplotlib.pyplot as plt

# For some configurations, this script plots the determined label errors in the original ground truth or a random subset of which.
# Label errors are cropped to a size of 100 x 100 pixels from the bbox center and stored as images in the folder "label_error_imgs".
# The script also generates the latex code to include the images with the filename as a caption.

# We generate the following images of label errors:
# - Most evident cases of overlooked pedestrians (Large objects of bbox height > 40 pixels, outside of dont care regions, annotated in validated GT with p > 0.8)
    # - all 63, sorted by soft label probability
# - Misfitting annotations of most evident cases (Same conditions as above)
    # - first 20 examples sorted ascending by IoU with val. annotations. Note that here many objects were overlooked and just intersect slightly with anothe GT annotation.
# - Small but unambiguous objects (Height < 40 pixels, annotated in validated GT with p > 0.8)
    # - random sample of 12 examples, set random seed to 0 for reproducibility
# - Ambiguous objects (annotated in validated GT with 0.5 < p < 0.8)
    # - random sample of 12 examples, set random seed to 0 for reproducibility

def filter_data_for_conditions(df):
    if filter_dont_care:
        df = df[df["ioa_with_dont_care"] < ioa_thresh_dont_care]
    if min_bbox_height:
        df = df[df["height"] >= min_bbox_height]
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

def read_kitti_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            # 2D bounding box: [left, top, right, bottom]
            bbox = list(map(float, parts[4:8]))
            boxes.append((cls, bbox))
    return boxes

def plot_and_store_label_error(image_id, pred_boxes=None, add_pred_boxes=None, scores_pred_boxes = None, scores_add_pred_boxes = None, filename=None, crop_size = 100):
    image_id = image_id.split('.')[0]  # Remove file extension if present
    image_path = f'{path_to_kitti}/image_2/{image_id}.png'
    label_path = f'{path_to_kitti}label_2/{image_id}.txt'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw ground truth boxes
    gt_boxes = read_kitti_labels(label_path)
    for cls, (left, top, right, bottom) in gt_boxes:
        if cls == "Pedestrian": # we only care about pedestrians
            pt1 = (int(left), int(top))
            pt2 = (int(right), int(bottom))
            cv2.rectangle(image, pt1, pt2, color=(0, 255, 0), thickness=2)

    # Draw predicted boxes (if provided)
    if pred_boxes is not None:
        for i, pred in enumerate(pred_boxes):
            xmin, ymin, xmax, ymax = pred[:4]
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            cv2.rectangle(image, pt1, pt2, color=(255, 0, 0), thickness=2)  # Red
            if scores_pred_boxes is not None:
                score = scores_pred_boxes[i]
                label = f" ({score:.4f})"
                cv2.putText(image, label, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)

            # Crop the image around the center of the bounding box
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            crop_x_min = max(center_x - crop_size, 0)
            crop_y_min = max(center_y - crop_size, 0)
            crop_x_max = min(center_x + crop_size, image.shape[1])
            crop_y_max = min(center_y + crop_size, image.shape[0])
            cropped_image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            # Display the cropped image
            plt.figure(figsize=(5, 5))
            plt.imshow(cropped_image)
            plt.axis('off')
            if filename:
                plt.savefig(filename, dpi = 100)
            plt.show()
            plt.close()

if __name__ == "__main__":

    # fixed parameters for evaluation
    iou_threshold_misfitting_box = 0.5  # IoU threshold for misfitting boxes (FP of predictions w.r.t. original GT)
    ioa_thresh_dont_care = 0.5  # IoU threshold for considering the intersection with dont care regions

    # paths to the files
    path_to_original_gt = "data/original_gt.csv"
    path_to_validated_gt = "data/validated_gt.csv"

    path_to_kitti = '/home/datasets_archive/KITTI/training/'


    ############ Load original and validated annotations ############

    # load the original ground truth and the dont care regions
    original_gt = pd.read_csv(path_to_original_gt)
    dont_care_regions = pd.read_csv("data/original_gt_dont_care_regions.csv")
    
    # load the validated ground truth
    validated_gt = pd.read_csv(path_to_validated_gt)

    ############ 1. & 2. Set parameters accordingly to infer most evident label errors ############
    
    # variable parameters for evaluation -> Changes data under consideration e.g. only larger objects or only objects with a certain probability in soft label annotation
    validated_gt_prob_threshold = 0.8  # Probability threshold for ground truth class pedestrian (0.5 and 0.8 available)
    min_bbox_height = 40 # minimal height of a bounding box to be considered as a pedestrian (according to KITTI Benchmark moderate and hard version). Filters all boxes.
    filter_dont_care = True

    ########### Matching Procedure as in compare_annotations.py ###########
    # filter validated_gt for the conditions set by the parameters such as dont care regions, small boxes, as well as the probability threshold for the validated GT
    validated_gt = compare_orig_and_val_gt(validated_gt, original_gt, dont_care_regions) # determine IoA scores
    validated_gt = validated_gt[validated_gt["probability"] >= validated_gt_prob_threshold]
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering
    validated_gt = filter_data_for_conditions(validated_gt) # filters out small boxes and dont care regions in validated GT
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    original_gt = compare_orig_and_val_gt(original_gt, original_gt, dont_care_regions) # determine IoA scores
    original_gt = filter_data_for_conditions(original_gt) # filters out small boxes and boxes in dont care regions in orig. GT
    original_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    #  Determine the number of label errors in the original GT based on the configuration of the validated GT.
    validated_gt = determine_label_errors_in_original_gt(original_gt, validated_gt, iou_threshold_misfitting_box)
    number_of_label_errors = len(validated_gt.query('label_error == True'))

    # print(f"Total number of label errors: {number_of_label_errors}", "(Overlooked: ", len(validated_gt.query('label_error_type == "overlooked pedestrian"')), ", Misfitting: ", len(validated_gt.query('label_error_type == "misfitting box"')), ")")

    # 1. Most evident cases of overlooked pedestrians
    # - Large objects of bbox height >= 40 pixels, outside of dont care regions, annotated in validated GT with p >= 0.8

    filename_map = {}

    label_errors = validated_gt.copy()

    # --- CONFIGURATION ---
    image_dir = "label_error_imgs/top_overlooked/" 
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    output_file = image_dir + "label_error_imgs_top_overlooked.tex"

    label_errors = label_errors[label_errors["label_error_type"] == "overlooked pedestrian"]
    label_errors = label_errors.sort_values(by="probability", ascending=False)
    label_errors.reset_index(drop=True, inplace=True)

    N = len(label_errors)
    print(f"Plotting {N} most evident cases of overlooked pedestrians to "+image_dir)

    for loop_index in range(N):
        index = label_errors.index[loop_index]

        label_error = label_errors.iloc[index][['xmin', 'ymin', 'xmax', 'ymax', 'probability']].values.reshape(1, -1)
        image_id = label_errors.iloc[index][['filename']].values[0]

        filename = f"label_error_{loop_index}.png"
        filename_map[filename] = image_id

        plot_and_store_label_error(image_id, pred_boxes=label_error, filename = image_dir + filename)


    # write latex code for inclusion in paper
    latex_lines = []
    images_per_row = 4

    for i, (img_name, original_name) in enumerate(filename_map.items()):
        if i % images_per_row == 0:
            latex_lines.append(r"\noindent")  # Start of a new row

        latex_lines.append(
            rf"""\begin{{minipage}}[t]{{0.24\linewidth}}
      \centering
      \includegraphics[width=\linewidth, , height=\linewidth, keepaspectratio]{{imgs/{image_dir}{img_name}}}
      \captionsetup{{labelformat=empty, hypcap=false}}
      \captionof{{figure}}{{{original_name}}}
    \end{{minipage}}"""
        )

        if (i + 1) % images_per_row == 0:
            latex_lines.append(r"\par")  # Space between rows with \smallskip

    # Save LaTeX code to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))


    # 2. Most evident cases of misfitting annotations
    # - Large objects of bbox height >= 40 pixels, outside of dont care regions, annotated in validated GT with p >= 0.8
    # Show first 20 examples sorted ascending by IoU with val. annotations. Note that here many objects were overlooked and just intersect slightly with another GT annotation.

    filename_map = {}

    # --- CONFIGURATION ---
    image_dir = "label_error_imgs/top_misfitting/" 
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    output_file = image_dir + "label_error_imgs_top_misfitting.tex"

    label_errors = validated_gt.copy()
    label_errors = label_errors[label_errors["label_error_type"] == "misfitting box"]
    label_errors = label_errors.sort_values(by="iou_with_original_gt", ascending=True)
    label_errors.reset_index(drop=True, inplace=True)

    N = 20
    print(f"Plotting {N} most evident cases of misfitting pedestrian annotations to "+image_dir)


    for loop_index in range(N):
        index = label_errors.index[loop_index]

        label_error = label_errors.iloc[index][['xmin', 'ymin', 'xmax', 'ymax', 'probability']].values.reshape(1, -1)
        image_id = label_errors.iloc[index][['filename']].values[0]

        filename = f"label_error_{loop_index}.png"
        filename_map[filename] = image_id

        if not loop_index in [11, 16, 19]:
            plot_and_store_label_error(image_id, pred_boxes=label_error, filename = image_dir + filename)
        else:
            plot_and_store_label_error(image_id, pred_boxes=label_error, filename = image_dir + filename, crop_size = 200) # larger crop for these examples after manual inspection


    # write latex code for inclusion in paper
    latex_lines = []
    images_per_row = 4

    for i, (img_name, original_name) in enumerate(filename_map.items()):
        if i % images_per_row == 0:
            latex_lines.append(r"\noindent")  # Start of a new row

        latex_lines.append(
            rf"""\begin{{minipage}}[t]{{0.24\linewidth}}
      \centering
      \includegraphics[width=\linewidth, , height=\linewidth, keepaspectratio]{{imgs/{image_dir}{img_name}}}
      \captionsetup{{labelformat=empty, hypcap=false}}
      \captionof{{figure}}{{{original_name}}}
    \end{{minipage}}"""
        )

        if (i + 1) % images_per_row == 0:
            latex_lines.append(r"\par")  # Space between rows with \smallskip

    # Save LaTeX code to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    ############ 3. & 4. Set less strict parameters ############
    # load the original ground truth and the dont care regions
    original_gt = pd.read_csv(path_to_original_gt)
    dont_care_regions = pd.read_csv("data/original_gt_dont_care_regions.csv")
    
    # load the validated ground truth
    validated_gt = pd.read_csv(path_to_validated_gt)
    
    # variable parameters for evaluation -> Changes data under consideration e.g. only larger objects or only objects with a certain probability in soft label annotation
    validated_gt_prob_threshold = 0.5  # Probability threshold for ground truth class pedestrian (0.5 and 0.8 available)
    min_bbox_height = 0 # minimal height of a bounding box to be considered as a pedestrian (according to KITTI Benchmark moderate and hard version). Filters all boxes.
    filter_dont_care = True

    ########### Matching Procedure as in compare_annotations.py ###########
    # filter validated_gt for the conditions set by the parameters such as dont care regions, small boxes, as well as the probability threshold for the validated GT
    validated_gt = compare_orig_and_val_gt(validated_gt, original_gt, dont_care_regions) # determine IoA scores
    validated_gt = validated_gt[validated_gt["probability"] >= validated_gt_prob_threshold]
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering
    validated_gt = filter_data_for_conditions(validated_gt) # filters out small boxes and dont care regions in validated GT
    validated_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    original_gt = compare_orig_and_val_gt(original_gt, original_gt, dont_care_regions) # determine IoA scores
    original_gt = filter_data_for_conditions(original_gt) # filters out small boxes and boxes in dont care regions in orig. GT
    original_gt.reset_index(drop=True, inplace=True)  # reset index after filtering

    #  Determine the number of label errors in the original GT based on the configuration of the validated GT.
    validated_gt = determine_label_errors_in_original_gt(original_gt, validated_gt, iou_threshold_misfitting_box)
    number_of_label_errors = len(validated_gt.query('label_error == True'))

    # print(f"Total number of label errors: {number_of_label_errors}", "(Overlooked: ", len(validated_gt.query('label_error_type == "overlooked pedestrian"')), ", Misfitting: ", len(validated_gt.query('label_error_type == "misfitting box"')), ")")

    # 3. Small but unambiguous objects
    # - Height < 40 pixels, annotated in validated GT with p >= 0.8
    # Consider a random sample of 12 examples, set random seed to 0 for reproducibility

    np.random.seed(0) # set seed because a random subset is considered

    filename_map = {}

    # --- CONFIGURATION ---
    image_dir = "label_error_imgs/small_but_confident/" 
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    output_file = image_dir + "label_error_imgs_small_but_confident.tex"

    label_errors = validated_gt.copy()
    label_errors = label_errors[label_errors["label_error_type"] == "overlooked pedestrian"]
    label_errors = label_errors[label_errors["ioa_with_dont_care"] < 0.5]
    label_errors = label_errors[label_errors["height"] < 40]
    label_errors = label_errors[label_errors["probability"] >= 0.8]
    label_errors.reset_index(drop=True, inplace=True)

    N = 12

    print(f"Plotting {N} random examples of label errors with small bounding boxes but high soft label probability to "+image_dir)


    indices = np.random.choice(label_errors.index, size = N, replace = False)

    for loop_index in indices:
        index = label_errors.index[loop_index]

        label_error = label_errors.iloc[index][['xmin', 'ymin', 'xmax', 'ymax', 'probability']].values.reshape(1, -1)
        image_id = label_errors.iloc[index][['filename']].values[0]

        filename = f"label_error_{loop_index}.png"
        filename_map[filename] = image_id

        plot_and_store_label_error(image_id, pred_boxes=label_error, filename = image_dir + filename)


    # write latex code for inclusion in paper
    latex_lines = []
    images_per_row = 4

    for i, (img_name, original_name) in enumerate(filename_map.items()):
        if i % images_per_row == 0:
            latex_lines.append(r"\noindent")  # Start of a new row

        latex_lines.append(
            rf"""\begin{{minipage}}[t]{{0.24\linewidth}}
      \centering
      \includegraphics[width=\linewidth, , height=\linewidth, keepaspectratio]{{imgs/{image_dir}{img_name}}}
      \captionsetup{{labelformat=empty, hypcap=false}}
      \captionof{{figure}}{{{original_name}}}
    \end{{minipage}}"""
        )

        if (i + 1) % images_per_row == 0:
            latex_lines.append(r"\par")  # Space between rows with \smallskip

    # Save LaTeX code to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    # 4. Ambiguous objects
    # - annotated in validated GT with 0.5 < p < 0.8
    # Consider a random sample of 12 examples, set random seed to 0 for reproducibility
    np.random.seed(0) 

    filename_map = {}

    # --- CONFIGURATION ---
    image_dir = "label_error_imgs/ambiguous/" 
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    output_file = image_dir + "label_error_imgs_ambiguous.tex"

    label_errors = validated_gt.copy()
    label_errors = label_errors[label_errors["label_error_type"] == "overlooked pedestrian"]
    label_errors = label_errors[label_errors["ioa_with_dont_care"] < 0.5]
    label_errors = label_errors[label_errors["probability"] < 0.8] # soft label probability filter for >= 0.5 is set already
    label_errors.reset_index(drop=True, inplace=True)

    N = 12

    print(f"Plotting {N} random examples of label errors with more ambiguous annotations to "+image_dir)


    indices = np.random.choice(label_errors.index, size = N, replace = False)

    for loop_index in indices:
        index = label_errors.index[loop_index]

        label_error = label_errors.iloc[index][['xmin', 'ymin', 'xmax', 'ymax', 'probability']].values.reshape(1, -1)
        image_id = label_errors.iloc[index][['filename']].values[0]

        filename = f"label_error_{loop_index}.png"
        filename_map[filename] = image_id

        plot_and_store_label_error(image_id, pred_boxes=label_error, filename = image_dir + filename)


    # write latex code for inclusion in paper
    latex_lines = []
    images_per_row = 4

    for i, (img_name, original_name) in enumerate(filename_map.items()):
        if i % images_per_row == 0:
            latex_lines.append(r"\noindent")  # Start of a new row

        latex_lines.append(
            rf"""\begin{{minipage}}[t]{{0.24\linewidth}}
      \centering
      \includegraphics[width=\linewidth, , height=\linewidth, keepaspectratio]{{imgs/{image_dir}{img_name}}}
      \captionsetup{{labelformat=empty, hypcap=false}}
      \captionof{{figure}}{{{original_name}}}
    \end{{minipage}}"""
        )

        if (i + 1) % images_per_row == 0:
            latex_lines.append(r"\par")  # Space between rows with \smallskip

    # Save LaTeX code to file
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))