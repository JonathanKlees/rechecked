import argparse
import logging
import os
import pickle
import uuid
from os.path import join
from tqdm import tqdm

import numpy as np
import pandas as pd
import shutil
import matplotlib
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_iou_matrix(gt_boxes, pred_boxes):
    """
    Compute IoU between each pair of GT and predicted boxes.
    gt_boxes: (N, 4), pred_boxes: (M, 4) -> returns (N, M) matrix
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))


    gt_boxes = np.array(gt_boxes)
    pred_boxes = np.array(pred_boxes)

    # Convert boxes from cx, cy, w, h to x1, y1, x2, y2
    gt_x1y1 = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
    gt_x2y2 = gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
    pred_x1y1 = pred_boxes[:, :2] - pred_boxes[:, 2:] / 2
    pred_x2y2 = pred_boxes[:, :2] + pred_boxes[:, 2:] / 2

    iou_matrix = np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]))

    for i, (gt_x1, gt_y1) in enumerate(gt_x1y1):
        gt_x2, gt_y2 = gt_x2y2[i]
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        for j, (pred_x1, pred_y1) in enumerate(pred_x1y1):
            pred_x2, pred_y2 = pred_x2y2[j]
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

            inter_x1 = max(gt_x1, pred_x1)
            inter_y1 = max(gt_y1, pred_y1)
            inter_x2 = min(gt_x2, pred_x2)
            inter_y2 = min(gt_y2, pred_y2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            union_area = gt_area + pred_area - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            iou_matrix[i, j] = iou

    return iou_matrix


def match_and_count(gt_bboxes,gt_scores, pred_bboxes, pred_scores,iou_thresh):
    """Greedy match predictions to GT and compute FN, FP counts."""
    if len(gt_bboxes) == 0:
        # No GT → all predictions are false positives
        return 0, len(pred_bboxes), [], [], []
    if len(pred_bboxes) == 0:
        # No predictions → all GTs are false negatives
        return len(gt_bboxes), 0, [], [], []

    iou_matrix = compute_iou_matrix(gt_bboxes, pred_bboxes)
    matched_gt = set()
    matched_pred = set()

    # Flatten IoU matrix and sort matches by IoU descending
    matches = [
        (i, j, iou_matrix[i, j])
        for i in range(iou_matrix.shape[0])
        for j in range(iou_matrix.shape[1])
        if iou_matrix[i, j] >= iou_thresh
    ]
    matches.sort(key=lambda x: -x[2])  # highest IoU first

    absolute_errors = []
    for gt_idx, pred_idx, iou in matches:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue  # already matched
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

        if gt_scores is not None:
            absolute_errors.append(abs(gt_scores[gt_idx] - pred_scores[pred_idx]))

    fn = len(gt_bboxes) - len(matched_gt)  # GTs not matched
    fp = len(pred_bboxes) - len(matched_pred)  # predictions not matched

    return fn, fp, absolute_errors, matched_gt, matched_pred


def detect_label_errors(num_preds, scores, matched_gt, matched_gt_pred, matched_org_gt, matched_org_gt_pred,cost_per_sample,is_labeling_strategy, threshold_validated_gt):
    label_errors_predictions = [""] * num_preds  # To store the type of label error (no, overlooked pedestrian or misfitting box)
    costs = [0.0] * num_preds  # costs to correct label error

    if num_preds == 0:
        # No predictions -> just missing, no correction possible
        return label_errors_predictions, costs

    for idx in range(num_preds):

        if idx in matched_gt_pred:
            # matched with validated gt
            matched_validated_gt = True

        else:
            # not matched with validated gt
            matched_validated_gt = False

        if idx in matched_org_gt_pred:
            # found match in original gt
            if matched_validated_gt:
                label_errors_predictions[idx] = "no_tp"  # found in validated gt and original

                # overwrites for labeling strategy
                if is_labeling_strategy:
                    # given labeling strategy label might be different
                    if scores[idx] >= threshold_validated_gt:
                        # applied cost yield the same result
                        pass
                    else:
                        # wrong label created
                        label_errors_predictions[idx] = "fn_correction"
            else:
                label_errors_predictions[idx] = "overlooked_fp"  # not found in validated gt but in original one
                # since it was proposed by method it is most likely overlooked fp

                # overwrites for labeling strategy
                if is_labeling_strategy:
                    # given labeling strategy label might be different
                    if scores[idx] >= threshold_validated_gt:
                        # different result
                        label_errors_predictions[idx] = "missed_fp"
                    else:
                        # same results
                        pass

            # for labeling strategy all samples need to be paid
            if is_labeling_strategy:
                costs[idx] = cost_per_sample
        else:

            if matched_validated_gt:
                label_errors_predictions[idx] = "fn_original_gt" # in original gt

                # overwrites for labeling strategy
                if is_labeling_strategy:
                    # given labeling strategy label might be different
                    if scores[idx] >= threshold_validated_gt:
                        # same results
                        pass
                    else:
                        # wrong label created
                        label_errors_predictions[idx] = "missed_fn"
            else:
                # meaning less no matches
                label_errors_predictions[idx] = "no_tn"  # found not in validated gt and original

                # overwrites for labeling strategy
                if is_labeling_strategy:
                    # given labeling strategy label might be different
                    if scores[idx] >= threshold_validated_gt:
                        # different result
                        label_errors_predictions[idx] = "fp_correction"
                    else:
                        # same results
                        pass

            # costs to check the quality of this proposed bounding box
            costs[idx] = cost_per_sample

    return label_errors_predictions, costs


if __name__ == "__main__":

    os.makedirs("plots", exist_ok=True)
    os.makedirs("cache_files", exist_ok=True)

    path_to_predictions = "data/predictions/all_predictions.pkl"

    default_costs = 15.59357
    # for all bounding boxes in cents
    offset_costs_combined = 39933
    offset_costs_bboxes = 15620
    offset_costs_keypoint_box = 24314

    validated_gt_thresholds = [0.5, 0.8]
    min_box_sizes = [0, 25]

    ########## DEFINE IDS ##########
    simple_names = {}
    
    # validated gt
    ids = []
    validated_gt_name = "Validated GT"
    simple_names[validated_gt_name] = validated_gt_name
    ids.append((validated_gt_name, default_costs, offset_costs_combined, True))

    # original gt
    original_gt_name = "Original GT"
    simple_names[original_gt_name] = original_gt_name
    ids.append((original_gt_name, default_costs, 0 , True))

    # cascadercnn loss
    comparison_name = "Instance-wise Loss Method"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0, False))

    # yolox
    comparison_name = "YOLOX"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0, False))
    # meta detect
    comparison_name = "YOLOX+MetaDetect"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0, False))
    # cleanlab
    comparison_name = "YOLOX+ObjectLab"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0, False))

    # cascadercnn
    comparison_name = "Cascade R-CNN"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0,  False))
    # meta detect
    comparison_name = "Cascade R-CNN+MetaDetect"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0,  False))
    # cleanlab
    comparison_name = "Cascade R-CNN+ObjectLab"
    simple_names[comparison_name] = comparison_name
    ids.append((comparison_name, default_costs, 0,  False))


    ## BBOX Generation Strategies
    box_gen = []
    # Keypoint + bbox
    comparison_name = "Keypoint-to-box Annotation"
    simple_names[comparison_name] = comparison_name
    box_gen.append((comparison_name, offset_costs_keypoint_box))

    #  direct bbox annotation
    comparison_name = "Direct Box Annotation"
    simple_names[comparison_name] = comparison_name
    box_gen.append((comparison_name, offset_costs_bboxes))

    # combined bboxes
    comparison_name = "Combined Box Annotation"
    simple_names[comparison_name] = comparison_name
    box_gen.append((comparison_name, offset_costs_combined))


    # LABELING STRATEGIES
    label_strats = []
    # Is correct
    label_strats.append(("Is pedestrian?", 4.7339))
    # person + walking?
    label_strats.append(("Is human \& stand/walk?", 7.1286))
    # person + walking? + ambigous ( non-grouped GT)
    label_strats.append(("Is human \& stand/walk? \& AR", 9.7318))

    # add all combinations of label and box strategy
    for box in box_gen:
        for label_strat in label_strats:
            ids.append((r'\shortstack[l]{' + box[0] +"+\\\\"+label_strat[0] + "}", label_strat[1], box[1], True))
            # add simple names
            if label_strat[0] == "Is pedestrian?":  
                simple_names[r'\shortstack[l]{' + box[0] +"+\\\\"+label_strat[0] + "}"] = box[0] + " Is pedestrian"
            if label_strat[0] == "Is human \& stand/walk?":
                simple_names[r'\shortstack[l]{' + box[0] +"+\\\\"+label_strat[0] + "}"] = box[0] + " Is human and stand or walk"
            if label_strat[0] == "Is human \& stand/walk? \& AR":
                simple_names[r'\shortstack[l]{' + box[0] +"+\\\\"+label_strat[0] + "}"] = box[0] + " Is human and stand or walk and AR"

    ############### Load data

    print(f"Loading data from {path_to_predictions}")

    with open(path_to_predictions, 'rb') as f:
        predictions_data = pickle.load(f) # Load data from the exported file

    bboxes = predictions_data["bboxes"]
    scores = predictions_data["scores"]
    mo_ids = predictions_data["mo_ids"]
    mo_id_to_mid = predictions_data["mo_id_to_mid"]

    ############## Start Evaluation

    ious_to_sweep = [0.1, 0.5]
    confidences_to_sweep = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    for validated_gt_threshold in validated_gt_thresholds:
        for min_box_size in min_box_sizes:
            print(f"Analyzing for validated gt threshold: {validated_gt_threshold} and min box size: {min_box_size}")

            cache_file = f'cache_files/all_comparisons_{validated_gt_threshold}_{min_box_size}.pkl'

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    print(f"Loaded cached dictionary from {cache_file}")
                    all_comparisons =  pickle.load(f)
            else:
                all_comparisons = {}

                for comparison_name, cost_per_sample, offset_costs, is_labeling_strategy in ids:
                    print(f"#### Analyzing {simple_names[comparison_name]}")


                    gt_bboxes = bboxes[validated_gt_name]
                    gt_scores = scores[validated_gt_name]
                    gt_mo_ids = mo_ids[validated_gt_name]
                    org_gt_bboxes = bboxes[original_gt_name]
                    org_gt_mo_ids = mo_ids[original_gt_name]
                    comparison_bboxes = bboxes[comparison_name]
                    comparison_scores = scores[comparison_name]
                    comparison_mo_ids = mo_ids[comparison_name]

                    # get hard labels based on threshold
                    gt_labels = [1 if score >= validated_gt_threshold else 0 for score in gt_scores]
                    gt_bboxes = [bbox for i, bbox in enumerate(gt_bboxes) if gt_labels[i] == 1] # filter down bboxes
                    gt_mo_ids =  [mo_id for i, mo_id in enumerate(gt_mo_ids) if gt_labels[i] == 1] # filter down media object ids
                    gt_scores =  [score for i, score in enumerate(gt_scores) if gt_labels[i] == 1] # filter down gt scores
                    
                    gt_media_ids = [mo_id_to_mid[mo_id] for mo_id in gt_mo_ids]
                    org_gt_media_ids = [mo_id_to_mid[mo_id] for mo_id in org_gt_mo_ids]
                    comparison_media_ids = [mo_id_to_mid[mo_id] for mo_id in comparison_mo_ids]
                    unique_media_ids = list(set(gt_media_ids + comparison_media_ids + org_gt_media_ids))

                    media_data = {}

                    for media_id in unique_media_ids:
                        # ensure of media and of correct size
                        gt_inds = [i for i, mid in enumerate(gt_media_ids) if mid == media_id and gt_bboxes[i][3] >= min_box_size]
                        org_gt_inds = [i for i, mid in enumerate(org_gt_media_ids) if mid == media_id  and org_gt_bboxes[i][3] >= min_box_size]
                        pred_inds = [i for i, mid in enumerate(comparison_media_ids) if mid == media_id  and comparison_bboxes[i][3] >= min_box_size]

                        gt_bboxes_img = [gt_bboxes[i] for i in gt_inds]
                        gt_scores_img = [gt_scores[i] for i in gt_inds]
                        org_gt_bboxes_img = [org_gt_bboxes[i] for i in org_gt_inds]

                        pred_bboxes_img = [comparison_bboxes[i] for i in pred_inds]
                        pred_scores_img = [comparison_scores[i] for i in pred_inds]
                        pred_moid_img = [comparison_mo_ids[i] for i in pred_inds]

                        # # Pre-sort predictions by score descending (optional for faster thresholding)
                        # pred_sorted = sorted(zip(pred_bboxes_img, pred_scores_img), key=lambda x: -x[1])

                        media_data[media_id] = (gt_bboxes_img, gt_scores_img, org_gt_bboxes_img, pred_bboxes_img, pred_scores_img,pred_moid_img)

                    # OWN SCORES

                    # add FPR, FNR, Costs, label errors


                    all_curves = {}  # {iou_thresh: (fpr_list, fnr_list)}

                    for iou_thresh in tqdm(ious_to_sweep):
                        (fpr_list, fnr_list, mae_list, cost_list,
                         percentage_match_list, label_error_list,
                         label_okay_list, label_overlooked_fp_list, label_error_missing_box_list,
                         total_gt_list, total_pred_list) = [], [], [], [], [], [], [], [], [], [], []

                        for pred_score_threshold in confidences_to_sweep:
                            # aggegration per sweep
                            total_fn, total_fp, total_gt, total_pred = 0, 0, 0, 0
                            absolute_errors, label_errors, label_okays, label_overlooked_fp, label_errors_different_labeling = [], 0,0,0,0
                            costs = []
                            label_error_moid_list = []

                            for media_id in unique_media_ids:
                                gt_bboxes_img, gt_scores_img, org_gt_bboxes_img, pred_bboxes_img, pred_scores_img, pred_moid_img = media_data[media_id]

                                # apply prediction score threshold
                                pred_bboxes_img = [bbox for k, bbox in enumerate(pred_bboxes_img) if
                                                   pred_scores_img[k] >= pred_score_threshold]
                                pred_moid_img = [moid for k, moid in enumerate(pred_moid_img) if
                                                 pred_scores_img[k] >= pred_score_threshold]
                                pred_scores_img = [score for k, score in enumerate(pred_scores_img) if
                                                   pred_scores_img[k] >= pred_score_threshold]


                                _, _, _, matched_org_gt, matched_org_gt_pred = (
                                    match_and_count(org_gt_bboxes_img, None,
                                                    pred_bboxes_img, pred_scores_img,
                                                    0.5))  # use realistic iou due to matching

                                fn, fp, ae, matched_gt, matched_gt_pred = (
                                    match_and_count(gt_bboxes_img,gt_scores_img,
                                                    pred_bboxes_img, pred_scores_img,
                                                    iou_thresh)) # use potentially low iou to find label information instead of actually labeling

                                total_fn += fn
                                total_fp += fp
                                total_gt += len(gt_bboxes_img)
                                total_pred += len(pred_bboxes_img)
                                absolute_errors.extend(ae)

                                # cost calculation and label errors
                                # with greedy gt matching
                                label_errors_raw, c = detect_label_errors(len(pred_bboxes_img),pred_scores_img, matched_gt, matched_gt_pred,
                                                      matched_org_gt, matched_org_gt_pred,
                                                      cost_per_sample, is_labeling_strategy and comparison_name != original_gt_name, validated_gt_threshold)

                                # determine percentage of label errors
                                labels, counts = np.unique(label_errors_raw, return_counts=True)
                                label_to_count = dict(zip(labels, counts))
                                label_error = label_to_count.get('fn_original_gt', 0)
                                label_okay = label_to_count.get('no_tn', 0) + label_to_count.get('no_tp', 0)

                                # get list of moids with label error
                                if comparison_name == validated_gt_name and pred_score_threshold == 0 and iou_thresh == 0.1:
                                    label_error_fn_moids = [moid for idx, moid in enumerate(pred_moid_img) if label_errors_raw[idx] == 'fn_original_gt']
                                    label_error_fn_indices = [idx for idx, moid in enumerate(pred_moid_img) if label_errors_raw[idx] == 'fn_original_gt']

                                costs.extend(c)
                                label_okays+=label_okay
                                label_errors += label_error
                                label_overlooked_fp += label_to_count.get('overlooked_fp', 0)
                                label_errors_different_labeling += label_to_count.get('fp_correction', 0) + label_to_count.get('fn_correction', 0) + label_to_count.get('missed_fn', 0) + label_to_count.get('missed_fp', 0)


                            fpr = total_fp / total_pred if total_pred > 0 else 0
                            fnr = total_fn / total_gt if total_gt > 0 else 0
                            mae = np.mean(absolute_errors) if len(absolute_errors) > 0 else -0.1
                            average_cost = np.mean(costs) if len(costs) > 0 else -0.1
                            sum_cost = np.sum(costs) + offset_costs if len(costs) > 0 else 0  + offset_costs

                            total_gt_list.append(total_gt)
                            total_pred_list.append(total_pred)

                            # print(pred_score_threshold, iou_thresh, "FP", fpr, total_fp, "FN", fnr, total_fn, "GT:", total_gt, "ComP:", total_pred)
                            fpr_list.append(fpr)
                            fnr_list.append(fnr)
                            mae_list.append(mae)
                            cost_list.append(sum_cost)
                            percentage_match_list.append(1-fpr)

                            le = label_errors # / total_gt if total_gt > 0 else 0
                            lo = label_okays #/ total_pred if total_pred > 0 else 0
                            e = label_overlooked_fp #/ total_pred if total_pred > 0 else 0
                            lem = label_errors_different_labeling #/ total_pred if total_pred > 0 else 0

                            label_error_list.append(le)
                            label_okay_list.append(lo)
                            label_overlooked_fp_list.append(e)
                            label_error_missing_box_list.append(lem)


                        all_curves[iou_thresh] = (cost_list, fpr_list, fnr_list, mae_list,
                                                  percentage_match_list, label_error_list,
                                                  label_okay_list, label_overlooked_fp_list, label_error_missing_box_list,
                                                  total_gt_list, total_pred_list)


                    # store all curves for per method comparison
                    all_comparisons[comparison_name] = all_curves


                with open(cache_file, 'wb') as f:
                    pickle.dump(all_comparisons, f)
                print(f"Cached dictionary to {cache_file}")

            ####
            ## Plots of all metrics for one specific iou
            ####
            def create_triplet_colormap(base_colors, lighten_amount=0.5, darken_amount=0.2):

                def lighten(c, amount=0.5):
                    rgb = np.array(mcolors.to_rgb(c))
                    return mcolors.to_rgba(rgb + (1 - rgb) * amount)

                def darken(c, amount=0.5):
                    rgb = np.array(mcolors.to_rgb(c))
                    return mcolors.to_rgba(rgb * (1 - amount))

                extended_colors = []
                for i, c in enumerate(base_colors):
                    if i == 5:
                        continue # skip brown
                    extended_colors.append(darken(c, darken_amount))  # Dark shade
                    extended_colors.append(c)  # Original
                    extended_colors.append(lighten(c, lighten_amount))  # Light shade

                return extended_colors


            # Choose 10 distinct base colors (from tab10)
            base_colors = plt.get_cmap('tab10').colors
            cmap = create_triplet_colormap(base_colors)
            num_colors = len(all_comparisons)
            linestyles = ["solid", "dashed", "dotted"]

            primary_index = 0

            # iterate over confidence threshold
            iou_threshold = 0.1
            # iterate over iou thresholds
            # iou_threshold = None
            confidence_threshold_index = 0

            names_metrics = ["Total Annotation Time [h]",
                             "False Positive Rate (FPR)",
                             "False Negative Rate (FNR)",
                             "Mean Absolute Error (MAE)",
                             "Matched with Validated GT [\%]",
                             "Found Label Errors - FN in Org. GT[\#]",
                             "Label Okay [\#]",
                             "Overlooked Label Errors - FP in Org. GT [\#]",
                             "Introduced Label Errors [\#]",
                             "Total GT", 
                             "Total Pred"]

            # reduce number of metrics to plot for better readability
            metrics_to_plot = ["Total Annotation Time [h]",
                            "Found Label Errors - FN in Org. GT[\#]",
                            "Introduced Label Errors [\#]"]

            for idx_metric, name in enumerate(names_metrics):
                # print("###" + name + "###")
                if idx_metric == primary_index:
                    continue  # no self comparison

                if name not in metrics_to_plot:
                    continue

                lines = []
                plt.figure(figsize=(6,4))

                for idx, (comparison_name, all_curves) in enumerate(all_comparisons.items()):

                    if comparison_name == "Original GT": # skip original gt in this visualization
                        continue

                    color = cmap[idx] # / (num_colors - 1))
                    ls = linestyles[idx % len(linestyles)]
                    if iou_threshold is not None:
                        metric_lists = all_curves[iou_threshold]
                    else:
                        # Iterate over all IOU keys and extract the metric tuple at `confidence_threshold_index`
                        num_metrics = len(all_curves[0.1])
                        metric_lists = tuple(
                            [all_curves[iou][i][confidence_threshold_index] for iou in sorted(all_curves.keys())]
                            for i in range(num_metrics)
                        )
                        print(metric_lists)
                        # primary index conversion from cents to € to hours
                    
                    if name == "Introduced Label Errors [\#]": # here, we display only the last value 
                        line = plt.plot([(x/100)/4.5 for x in metric_lists[primary_index]], metric_lists[idx_metric], marker='o', ms =2.5, ls = ls, lw = 0.75, color=color,
                                label=comparison_name)
                    else:
                        line = plt.plot([(x/100)/4.5 for x in metric_lists[primary_index]], metric_lists[idx_metric], marker='o', ms = 2.5, ls = ls, lw = 0.75, color=color,
                                label=comparison_name)
                        plt.xticks([0,50,100,150,200])
                    
                    lines.append(line)

                    print(f"{simple_names[comparison_name]}@{confidences_to_sweep[confidence_threshold_index]:0.02f}@{iou_threshold:0.02f}:"
                          f"{metric_lists[idx_metric][confidence_threshold_index]:0.02f}")

                    # Highlight lowest confidence with a cross
                    plt.scatter((metric_lists[primary_index][0]/ 100) / 4.5, metric_lists[idx_metric][0],
                                color='black', marker='x', s = 15, label=None, zorder = 10)

                plt.xlabel(names_metrics[primary_index])
                plt.ylabel(names_metrics[idx_metric])
                # plt.legend(title="Method")
                plt.grid(True)
                plt.tight_layout()
                # plt.show()
                plt.savefig(f"plots/{validated_gt_threshold}_{min_box_size}_{name.split('[')[0]}.png", bbox_inches='tight', dpi=300)

            # Create a new figure just for the legend
            fig_legend = plt.figure(figsize=(3, 1))  # adjust size as needed
            handles =[line[0] for line in lines]
            labels=[name for name in all_comparisons.keys() if not name == "Original GT"]

            leg = fig_legend.legend(
                handles=handles,
                labels=labels,
                loc='center',
                frameon=False
            )

            fig_legend.savefig("plots/legend.png", bbox_inches='tight', dpi=300)

    print("#"*10, "Plots stored under plots/", "#"*10)