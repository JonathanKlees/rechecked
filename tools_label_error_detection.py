import os
import pandas as pd
import numpy as np
from torchvision.ops import nms, box_iou
import torch
import cv2
import matplotlib.pyplot as plt
import ast
import json

import sklearn
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, RocCurveDisplay

def construct_original_gt(path_to_kitti, split, filter_small_boxes, min_bbox_height):
    """
    Constructs the original ground truth DataFrame from the KITTI dataset.
    
    Args:
        path_to_kitti (str): Path to the directory containing KITTI data.
        split (dict) : containing the filenames of the val split.
        filter_small_boxes (bool): Whether to filter out boxes that are too small.
        min_bbox_height (int): Minimum height of a bounding box to be considered valid.
        
    Returns:
        pd.DataFrame: DataFrame containing the original ground truth data.
    """
    annotations = []
    label_dir = os.path.join(path_to_kitti, 'label_2')  # KITTI label directory

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = parts[0]  # Class name
                    
                    if label_file.split(".")[0] in split["val_imgs"]: # only val split
                        if cls == "Pedestrian":  # Filter for pedestrians
                            xmin, ymin, xmax, ymax = map(float, parts[4:8])
                            annotations.append({
                                'filename': label_file.replace('.txt', '.png'),
                                'xmin': xmin,
                                'ymin': ymin,
                                'xmax': xmax,
                                'ymax': ymax,
                                "class_name": cls,
                                'class': 2  # Store the class in a new column
                            })

    original_gt = pd.DataFrame(annotations)

    original_gt["height"] = original_gt["ymax"] - original_gt["ymin"]
    if filter_small_boxes:
        print(f"Filtering out {len(original_gt[original_gt['height'] < min_bbox_height])} boxes with height < {min_bbox_height} pixels")
        original_gt = original_gt[original_gt["height"] >= min_bbox_height]  # Filter out boxes that are too small
        original_gt.reset_index(drop=True, inplace=True)  # Reset the index
    return original_gt

def construct_dont_care_regions(path_to_kitti, split):
    """
    Constructs the original ground truth DataFrame from KITTI .
    
    Args:
        path_to_kitti (str): Path to the directory containing KITTI data.
        split (dict) : containing the filenames of the val split.
        
    Returns:
        pd.DataFrame: DataFrame containing dont care regions of the original ground truth data.
    """
    annotations = []
    label_dir = os.path.join(path_to_kitti, 'label_2')  # KITTI label directory

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = parts[0]  # Class name
                    
                    if label_file.split(".")[0] in split["val_imgs"]: # only val split
                        if cls == "DontCare":  # Filter for Dont care regions
                            xmin, ymin, xmax, ymax = map(float, parts[4:8])
                            annotations.append({
                                'filename': label_file.replace('.txt', '.png'),
                                'xmin': xmin,
                                'ymin': ymin,
                                'xmax': xmax,
                                'ymax': ymax,
                                "class_name": cls,
                                'class': -1  # Store the class in a new column
                            })

    dont_care_regions = pd.DataFrame(annotations)

    return dont_care_regions

# def construct_original_gt(path_to_kitti_val_csv_files, filter_small_boxes, min_bbox_height):
#     """
#     Constructs the original ground truth DataFrame from KITTI validation CSV files.
    
#     Args:
#         path_to_kitti_val_csv_files (str): Path to the directory containing KITTI validation CSV files.
#         filter_small_boxes (bool): Whether to filter out boxes that are too small.
#         min_bbox_height (int): Minimum height of a bounding box to be considered valid.
        
#     Returns:
#         pd.DataFrame: DataFrame containing the original ground truth data.
#     """
#     filenames = []
#     xmins = []
#     ymins = []
#     xmaxs = []
#     ymaxs = []

#     for image_id in os.listdir(path_to_kitti_val_csv_files):
#         if image_id.endswith(".csv"):
#             df = pd.read_csv(path_to_kitti_val_csv_files + image_id)
#             df = df[df['category_idx'] == 2]  # Filter for pedestrians (category_idx = 2)
#             filenames.extend([(image_id[:-7]+ ".png")]  * len(df))  # Remove _gt.csv extension and add .png
#             xmins.extend(df['xmin'].values)
#             ymins.extend(df['ymin'].values)
#             xmaxs.extend(df['xmax'].values)
#             ymaxs.extend(df['ymax'].values)

#     original_gt = pd.DataFrame({
#         'filename': filenames,
#         'xmin': xmins,
#         'ymin': ymins,
#         'xmax': xmaxs,
#         'ymax': ymaxs
#     })

#     original_gt["height"] = original_gt["ymax"] - original_gt["ymin"]
#     if filter_small_boxes:
#         print(f"Filtering out {len(original_gt[original_gt['height'] < min_bbox_height])} boxes with height < {min_bbox_height} pixels")
#         original_gt = original_gt[original_gt["height"] >= min_bbox_height]  # Filter out boxes that are too small
#         original_gt.reset_index(drop=True, inplace=True)  # Reset the index
#     return original_gt

def prepare_data(object_detector, object_detector_score_threshold, filter_small_boxes, min_bbox_height, nms_threshold):
    """
    Prepare the data for label error detection.
    
    Parameters:
    - object_detector: Name of the object detector used (e.g., "cascadercnn").
    - object_detector_score_threshold: Score threshold previously applied for filtering predictions (part of CSV name).
    - filter_small_boxes: Whether to filter out boxes that are too small. -> min_bbox_height is used for this.
    - nms_threshold: Threshold for Non-Maximum Suppression (None to disable NMS).
    
    Returns:
    - DataFrame with predictions and their scores.
    """
    csv_path = f"results/{object_detector}_md_results(t={object_detector_score_threshold}).csv"
    df = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"])
    df.columns = ["filename", 'xmin', 'ymin', 'xmax', 'ymax', 'score', 'dataset_box_id', 'target', 'prediction']

    df["class"] = [0]*len(df) # all predictions are for the same class "pedestrians"
    df["height"] = df["ymax"] - df["ymin"]
    if filter_small_boxes:
        print(f"Filtering out {len(df[df['height'] < min_bbox_height])} boxes with height < {min_bbox_height} pixels")
        df = df[df["height"] >= min_bbox_height]  # Filter out boxes that are too small
        df.reset_index(drop=True, inplace=True)  # Reset the index
    
    # proposed label errors do not overlap much due to NMS being applied in the object detector already.
    #  Here, we could further reduce this threshold if we wanted to.
    nms_df = perform_nms_on_dataframe(df, nms_threshold)
    print(f"Number of Boxes: {len(df)}", f", After NMS: {len(nms_df)}" )
    
    return nms_df

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

        for (image_id, cls), group in df.groupby(['filename', 'class']):
            boxes = torch.tensor(group[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
            scores = torch.tensor(group[score_column].values, dtype=torch.float32)
            
            if len(boxes) == 0:
                continue
            
            keep = nms(boxes, scores, threshold)
            kept_rows.append(group.iloc[keep.numpy()])

        return pd.concat(kept_rows).reset_index(drop=True)

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

    

def construct_cleanlab_input(path_to_kitti, split, path_to_predictions, n_classes=8, thresh=0.01):
    """
    Constructs the input for cleanlab from the KITTI validation dataset and predictions.
    
    Parameters:
    - path_to_kitti: Path to the directory containing the KITTI dataset.
    - path_to_predictions: Path to the JSON file containing the predictions.
    - n_classes: Number of classes in the dataset (default is 8).
    - thresh: Score threshold for filtering predictions (default is 0.01).
    
    Returns:
    - labels: List of dictionaries containing filenames, bounding boxes, and labels.
    - predictions: List of predictions in the format required by cleanlab.
    """
    
    # Ensure the directory exists
    if not os.path.exists(path_to_kitti):
        raise FileNotFoundError(f"The directory {path_to_kitti} does not exist.")
    
    # construct the original dataset in a suitable format for cleanlab
    labels = []

    label_dir = os.path.join(path_to_kitti, 'label_2')  # KITTI label directory

    for label_file in os.listdir(label_dir):
        label = dict()
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as f:
                bboxes = []
                for line in f:
                    parts = line.strip().split()
                    cls = parts[0]  # Class name
                    
                    if label_file.split(".")[0] in split["val_imgs"]: # only val split
                        if cls == "Pedestrian":  # Filter for pedestrians
                            xmin, ymin, xmax, ymax = map(float, parts[4:8])
                            bboxes.append([xmin, ymin, xmax, ymax])
                label['filename'] = label_file.replace('.txt', '.png')

                if len(bboxes) == 0:  # No pedestrians in this image
                    label["bboxes"] = np.empty((0, 4), dtype=np.float32)
                    label["labels"] = np.array([])
                else:
                    label["bboxes"] = np.array(bboxes, dtype=np.float32)
                    label["labels"] = np.array([2]*len(bboxes))  # All labels are pedestrian (category_idx = 2)
                
                labels.append(label)



    # for image_id in os.listdir(path_to_kitti_val_csv_files):
    #     if image_id.endswith(".csv"):
    #         label = dict()
    #         df = pd.read_csv(path_to_kitti_val_csv_files + image_id)
    #         df = df[df['category_idx'] == 2]  # Filter for pedestrians (category_idx = 2)
    #         label["filename"] = (image_id[:-7]+ ".png") # Remove _gt.csv extension and add .png
    #         xmins = df['xmin'].values
    #         ymins = df['ymin'].values
    #         xmaxs = df['xmax'].values
    #         ymaxs = df['ymax'].values

    #         if len(df) == 0:  # No pedestrians in this image
    #             label["bboxes"] = np.empty((0, 4), dtype=np.float32)
    #             label["labels"] = np.array([])
    #         else:
    #             boxes = []
    #             for i in range(len(df)): # add bboxes to label
    #                 boxes.append([xmins[i], ymins[i], xmaxs[i], ymaxs[i]])
    #             label["bboxes"] = np.array(boxes, dtype=np.float32)
    #             label["labels"] = np.array([2]*len(df))  # All labels are pedestrian (category_idx = 2)

    #         labels.append(label)

    empty_bbox = np.empty((0, 5), dtype=np.float32)
    empty_prediction = np.array([empty_bbox] * n_classes, dtype=object)

    # gather predictions for each image in the dataset with a score above the threshold

    # predictions are in the format: [empty_bbox, empty_bbox, bboxes_and_scores, empty_bbox, empty_bbox, empty_bbox, empty_bbox, empty_bbox]
    # where bboxes_and_scores is an array of shape (n_bboxes, 5) with each bbox as [xmin, ymin, xmax, ymax, score]
    # and empty_bbox is an empty array of shape (0, 5) for the other classes
    # this is because the predictions are always for the class pedestrian (category_idx = 2)

    predictions = []

    preds = json.load(open(path_to_predictions, "r"))

    for i in range(len(labels)): # gather predictions for each img of the dataset
        filename_to_find = labels[i]['filename']
        matching_pred = next((pred for pred in preds if pred['filename'] == filename_to_find), None)

        if matching_pred: # predictions are always for the class pedestrian so we gather the bboxes and scores
            bboxes = np.array(matching_pred['bboxes'], dtype=np.float32)
            scores = np.array(matching_pred['scores'], dtype=np.float32)
            
            # Filter bboxes based on scores above the threshold
            valid_indices = scores > thresh
            bboxes = bboxes[valid_indices]
            scores = scores[valid_indices]
            
            prediction = [empty_bbox] * n_classes  # Initialize with empty arrays for each class
            prediction[2] = np.hstack((bboxes, scores[:, np.newaxis]))

            predictions.append(np.array(prediction, dtype = object))
        else:
            # If no matching prediction is found, append an empty array
            predictions.append(empty_prediction)

    return labels, predictions

# def construct_cleanlab_input(path_to_kitti_val_csv_files, path_to_predictions, n_classes=8, thresh=0.01):
#     """
#     Constructs the input for cleanlab from the KITTI validation dataset and predictions.
    
#     Parameters:
#     - path_to_kitti_val_csv_files: Path to the directory containing the KITTI validation CSV files.
#     - path_to_predictions: Path to the JSON file containing the predictions.
#     - n_classes: Number of classes in the dataset (default is 8).
#     - thresh: Score threshold for filtering predictions (default is 0.01).
    
#     Returns:
#     - labels: List of dictionaries containing filenames, bounding boxes, and labels.
#     - predictions: List of predictions in the format required by cleanlab.
#     """
    
#     # Ensure the directory exists
#     if not os.path.exists(path_to_kitti_val_csv_files):
#         raise FileNotFoundError(f"The directory {path_to_kitti_val_csv_files} does not exist.")
    
#     # construct the original dataset in a suitable format for cleanlab
#     labels = []
#     for image_id in os.listdir(path_to_kitti_val_csv_files):
#         if image_id.endswith(".csv"):
#             label = dict()
#             df = pd.read_csv(path_to_kitti_val_csv_files + image_id)
#             df = df[df['category_idx'] == 2]  # Filter for pedestrians (category_idx = 2)
#             label["filename"] = (image_id[:-7]+ ".png") # Remove _gt.csv extension and add .png
#             xmins = df['xmin'].values
#             ymins = df['ymin'].values
#             xmaxs = df['xmax'].values
#             ymaxs = df['ymax'].values

#             if len(df) == 0:  # No pedestrians in this image
#                 label["bboxes"] = np.empty((0, 4), dtype=np.float32)
#                 label["labels"] = np.array([])
#             else:
#                 boxes = []
#                 for i in range(len(df)): # add bboxes to label
#                     boxes.append([xmins[i], ymins[i], xmaxs[i], ymaxs[i]])
#                 label["bboxes"] = np.array(boxes, dtype=np.float32)
#                 label["labels"] = np.array([2]*len(df))  # All labels are pedestrian (category_idx = 2)

#             labels.append(label)

#     empty_bbox = np.empty((0, 5), dtype=np.float32)
#     empty_prediction = np.array([empty_bbox] * n_classes, dtype=object)

#     # gather predictions for each image in the dataset with a score above the threshold

#     # predictions are in the format: [empty_bbox, empty_bbox, bboxes_and_scores, empty_bbox, empty_bbox, empty_bbox, empty_bbox, empty_bbox]
#     # where bboxes_and_scores is an array of shape (n_bboxes, 5) with each bbox as [xmin, ymin, xmax, ymax, score]
#     # and empty_bbox is an empty array of shape (0, 5) for the other classes
#     # this is because the predictions are always for the class pedestrian (category_idx = 2)

#     predictions = []

#     preds = json.load(open(path_to_predictions, "r"))

#     for i in range(len(labels)): # gather predictions for each img of the dataset
#         filename_to_find = labels[i]['filename']
#         matching_pred = next((pred for pred in preds if pred['filename'] == filename_to_find), None)

#         if matching_pred: # predictions are always for the class pedestrian so we gather the bboxes and scores
#             bboxes = np.array(matching_pred['bboxes'], dtype=np.float32)
#             scores = np.array(matching_pred['scores'], dtype=np.float32)
            
#             # Filter bboxes based on scores above the threshold
#             valid_indices = scores > thresh
#             bboxes = bboxes[valid_indices]
#             scores = scores[valid_indices]
            
#             prediction = [empty_bbox] * n_classes  # Initialize with empty arrays for each class
#             prediction[2] = np.hstack((bboxes, scores[:, np.newaxis]))

#             predictions.append(np.array(prediction, dtype = object))
#         else:
#             # If no matching prediction is found, append an empty array
#             predictions.append(empty_prediction)

#     return labels, predictions

def draw_boxes(image_path, label_path, pred_boxes=None, add_pred_boxes=None, scores_pred_boxes = None, scores_add_pred_boxes = None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw ground truth boxes
    gt_boxes = read_kitti_labels(label_path)
    for cls, (left, top, right, bottom) in gt_boxes:
        if cls == "Pedestrian": # we only care about pedestrians
            pt1 = (int(left), int(top))
            pt2 = (int(right), int(bottom))
            cv2.rectangle(image, pt1, pt2, color=(0, 255, 0), thickness=2)  # Green
            # cv2.putText(image, "GT", (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5, (0, 255, 0), 1)

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
                
    # Draw additional set of boxes (if provided)
    if add_pred_boxes is not None:
        for i, pred in enumerate(add_pred_boxes):
            xmin, ymin, xmax, ymax = pred[:4]
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            cv2.rectangle(image, pt1, pt2, color=(0, 0, 255), thickness=2)
            if scores_add_pred_boxes is not None:
                score = scores_add_pred_boxes[i]
                label = f" ({score:.4f})"
                cv2.putText(image, label, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1)

    plt.figure(figsize=(15, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def iou_from_df(df, i, j, df2 = None):
    """
    Compute the IoU between two boxes in a DataFrame using their row indices. Or pass a second DataFrame to compare boxes from two different DataFrames (assuming first index maps to first df).
    """
    if df2 is not None: # from two different DataFrames

        if df.loc[i, 'filename'] != df2.loc[j, 'filename']:
            print("Boxes are on different images, returning 0.")
            return 0.0
        else:
            box1 = np.array(df.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']].values, dtype = float)
            box2 = np.array(df2.iloc[j][['xmin', 'ymin', 'xmax', 'ymax']].values, dtype = float)

            box1 = torch.tensor(box1, dtype=torch.float32).unsqueeze(0)
            box2 = torch.tensor(box2, dtype=torch.float32).unsqueeze(0)

            iou = box_iou(box1, box2)[0, 0].item()
            return iou
        
    else:
        if df.loc[i, 'filename'] != df.loc[j, 'filename']:
            print("Boxes are on different images, returning 0.")
            return 0.0  # Different images → invalid comparison
        
        box1 = np.array(df.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']].values, dtype = float)
        box2 = np.array(df.iloc[j][['xmin', 'ymin', 'xmax', 'ymax']].values, dtype = float)

        box1 = torch.tensor(box1, dtype=torch.float32).unsqueeze(0)  # shape: (1, 4)
        box2 = torch.tensor(box2, dtype=torch.float32).unsqueeze(0)  # shape: (1, 4)

        iou = box_iou(box1, box2)[0, 0].item()
        return iou
    

def plot_roc_curve(df, score_col = "prediction", method="MetaDetect", object_detector = "cascadercnn"):
    
    y_true = df["TP"].values.astype(int)  # Convert boolean to int (0 or 1)
    y_scores = df[score_col].values  # Use the predicted scores (probabilities)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    fig = plt.figure()
    ax = plt.gca()

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax = ax)
    ax.get_legend().remove()

    AUROC = sklearn.metrics.roc_auc_score(y_true, y_scores)
    plt.annotate(
        'AUROC: %.4f' % AUROC,
        xy=(0.55, 0.25), xycoords='axes fraction',
        bbox=dict(boxstyle="round", fc="0.9", ec="teal")
    )
    plt.plot([0, 1], [0, 1], color='grey', ls="dashed")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.savefig(f"plots/ROC_{method}_{object_detector}.png", bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(df, score_col = "prediction", method="MetaDetect", object_detector = "cascadercnn"):
    
    y_true = df["TP"].values.astype(int)  # Convert boolean to int (0 or 1)
    y_scores = df[score_col].values  # Use the predicted scores (probabilities)
    
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_scores)

    fig = plt.figure()
    ax = plt.gca()
    prc_display = sklearn.metrics.PrecisionRecallDisplay(precision = precision, recall = recall).plot(ax = ax)

    AP = sklearn.metrics.average_precision_score(y_true, y_scores)
    ax.annotate(
    'AP: %.4f'%AP,
    xy=(0.05, 0.1), xycoords='axes fraction',
    bbox=dict(boxstyle="round", fc="0.9", ec="teal")
    )
    ax.hlines(len(df.query("TP == True"))/len(df),0,1, color = 'grey', ls = "dashed")
    plt.grid()
    plt.savefig(f"plots/PRC_{method}_{object_detector}.png", bbox_inches='tight')
    plt.show()

def f_1_score_plot(df, score_col = "prediction", step = 1, method="MetaDetect", object_detector = "cascadercnn"):
    # assuming df is sorted by the score col and contains a column "TP" indicating actual label errors out of the proposed ones.
    # considers increasingly large numbers of label error proposals and evaluates the F_1 score at the optimal threshold.
    X = []
    F_scores = []
    for n in range(step, len(df), step):
        X.append(n)
        df_n = df[:n]
        # consider all possible thresholds and the return the optimal F_1 score
        y_true = df_n["TP"].values.astype(int)  # Convert boolean to int (0 or 1)
        y_scores = df_n[score_col].values  # Use the predicted scores (probabilities)

        prec, rec, thresholds = precision_recall_curve(y_true, y_scores)

        # Compute F1 score for each threshold
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)

        # # Ignore last precision/recall value (not associated with a threshold)
        # f1_scores = f1_scores[:-1]

        # Find threshold with maximum F1
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)
        F_scores.append(best_f1)
        
    plt.figure()
    ax = plt.gca()
    ax.plot(X, F_scores)
    plt.xlabel("Number of proposed label errors")
    plt.ylabel("maximal $F_1$-Score")
    #plt.plot([X[0], X[-1]], [Y[0], Y[-1]], linestyle='--', color='gray', label='random guess')
    plt.grid()
    plt.savefig(f"plots/max_F_scores_{method}_{object_detector}.png", bbox_inches='tight')
    plt.show()

def analysis_plots(df, score_col = "prediction", method="MetaDetect", object_detector = "cascadercnn"):
    """
    Generate analysis plots for the proposed label errors.
    
    Parameters:
    - df: DataFrame containing the proposed label errors.
    - score_col: Column name for the predicted scores.
    - method: Method used for the analysis (e.g., "MetaDetect").
    - object_detector: Name of the object detector used (e.g., "cascadercnn").
    """

    X = []
    P_list = []
    N_list = []
    TP_list = []
    TN_list = []
    step = 1
    for n in range(step, len(df) + step, step):
        X.append(n)
        df_n = df[:n]
        TP = len(df_n[df_n['TP'] == True])
        TP_list.append(TP)
        TN_list.append(0) # We always predict label errors, so there are no true negatives in this case
        P_list.append( len(df_n[df_n['TP'] == True]) )
        N_list.append( len(df_n[df_n['TP'] == False]) )
        

    FN_list = [P_list[i] - TP_list[i] for i in range(len(TP_list))]  # False Negatives: original label errors that were not detected
    FP_list = [N_list[i] - TN_list[i] for i in range(len(TN_list))]  # False Positives: proposed label errors that are not in the original GT
    # Here, FP = N always since there are no negative predictions i.e. TN = 0
    F_scores = [2 * TP / (2 * TP + FN + FP) for TP, FN, FP in zip(TP_list, FN_list, FP_list)]  # F-score: harmonic mean of precision and recall

    Y = TP_list
    plt.figure()
    ax = plt.gca()
    ax.plot(X, Y)
    plt.xlabel("Number of proposed label errors")
    plt.ylabel("Number of true positives")
    plt.plot([X[0], X[-1]], [Y[0], Y[-1]], linestyle='--', color='gray', label='random guess')
    plt.grid()
    plt.savefig(f"plots/TPs_{method}_{object_detector}.png", bbox_inches='tight')
    plt.show()

    tpr = [TP_list[i] / X[i] for i in range(len(X))]
    plt.figure()
    ax = plt.gca()
    ax.plot(X, tpr)
    plt.xlabel("Number of proposed label errors")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.plot([X[0], X[-1]], [tpr[-1], tpr[-1]], linestyle='--', color='gray', label='random guess')
    plt.savefig(f"plots/TPR_{method}_{object_detector}.png", bbox_inches='tight')
    plt.show()

    Y = F_scores
    plt.figure()
    ax = plt.gca()
    ax.plot(X, Y)
    plt.xlabel("Number of proposed label errors")
    plt.ylabel("$F_1$-Score")
    #plt.plot([X[0], X[-1]], [Y[0], Y[-1]], linestyle='--', color='gray', label='random guess')
    plt.grid()
    plt.savefig(f"plots/F_scores_{method}_{object_detector}.png", bbox_inches='tight')
    plt.show()

    f_1_score_plot(df, score_col = score_col, step = step, method=method, object_detector = object_detector)

    plot_precision_recall_curve(df, score_col = score_col, method=method, object_detector = object_detector)
    plot_roc_curve(df, score_col = score_col, method=method, object_detector = object_detector)