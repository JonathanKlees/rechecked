import argparse
import logging
import os
import pickle
import uuid
from os.path import join

from hari_client import Config
from hari_client import HARIClient
from hari_client.utils.download import collect_media_and_attributes

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from plot_layout import set_plot_layout
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

def detect_label_errors(org_gt_bboxes, pred_bboxes, matched_preds,iou_thresh,cost_per_sample):
    label_error_type = [""] * len(
        pred_bboxes)  # To store the type of label error (no, overlooked pedestrian or misfitting box)
    label_errors = 0  # debugging variable
    costs = [0.0] * len(pred_bboxes)  # costs to correct label error

    if len(pred_bboxes) == 0:
        # No predictions -> just missing, no correction possible
        return label_error_type, costs


    iou_matrix = compute_iou_matrix(org_gt_bboxes, pred_bboxes)
    for idx, pred_bbox in enumerate(pred_bboxes):

        if idx in matched_preds:
            # matched with validated gt
            matched_validated_gt = True
        else:
            # not matched with validated gt
            matched_validated_gt = False

        # try to find match with original gt
        max_iou = max(iou_matrix[ :, idx]) if len(iou_matrix[ :, idx]) > 0 else 0 # maximum iou with gt boxes
        if max_iou < iou_thresh:
            if matched_validated_gt:
                # not matching original gt but in validated gt
                label_errors += 1
                if max_iou < 0.1:
                    label_error_type[idx] = "overlooked_pedestrian" # in original gt
                else:
                    # If the IoU is below the threshold but not zero, we consider it a misfitting box
                    label_error_type[idx] = "misfitting_box" # in original gt
                costs[idx] = cost_per_sample # costs to label one bbox to highest quality
            else:
                # meaning less no matches
                label_error_type[idx] = "no_tn"  # found not in validated gt and original
                # costs but no benefit
                costs[idx] = cost_per_sample
        else:
            if matched_validated_gt:
                label_error_type[idx] = "no_tp" # found in validated gt and original
                # no costs
            else:
                label_error_type[idx] = "overlooked_fp" # not found in validated gt but in original one

            # we would normally not check if there is a match with the ground truth
            # we expect no errors
            # however especially if the prediction is the original gt we might want to check it
            # costs[idx] = cost_per_sample


    return label_error_type, costs


def detect_label_errorsv2(num_preds, scores, matched_gt, matched_gt_pred, matched_org_gt, matched_org_gt_pred,cost_per_sample,is_labeling_strategy, threshold_validated_gt):
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

        # if an original gt was not matched it should most likely be checked to be a fp
        # these costs would totally mess up the calculation for predictions
        # leave them out for now
        # Reason:
        # we know the fp rate of the original gt and the percentage of overlooked fp,
        # -> thus also the non overlooked ones
        # the costs for each gt box is the same und thus can be seen as linear to the difference of overlooked fp - fp rate


    return label_errors_predictions, costs



if __name__ == "__main__":

    # OUTPUT
    cache_directory_name = "reliable"


    default_costs = 15.59357
    # for all bounding boxes in cents
    offset_costs_combined = 39933
    offset_costs_bboxes = 15620
    offset_costs_keypoint_box = 24314


    # DEFINE IDS
    ids = []
    dataset_id: uuid.UUID = uuid.UUID("ca5870ff-17bc-4ed1-bf1e-f86283c6341e")
    validated_gt_name = "Validated GT"
    # v1
    validated_gt_subset_id = uuid.UUID("d112d09e-780c-4b2b-a37a-2331b2e9478d")
    validated_gt_attribute_id: uuid.UUID = uuid.UUID("85f54ed9-db72-489f-8c64-63e5ce227c3d")
    # v2 - alignment with non-grouped bboxes, should result in better alignment and quality of bboxes
    validated_gt_subset_id = uuid.UUID("96b6bbd7-ee61-46ab-9dec-14c3cb43a15d")
    validated_gt_attribute_id: uuid.UUID = uuid.UUID("430f8846-0df4-4789-9146-205b063896b4")
    # v3 fix bug in soft label calculation
    validated_gt_attribute_id: uuid.UUID = uuid.UUID("d27c0b61-a8f5-4a42-a7de-37f2d8938c2d")
    # v5 fix bug in missing bounding boxes, + add weight for cant solve
    validated_gt_subset_id = uuid.UUID("ae598348-fefe-47cc-91fb-c7f2f182c5e8")
    validated_gt_attribute_id: uuid.UUID = uuid.UUID("09d755a8-1f67-4f00-82eb-3429773d4520")

    ids.append((validated_gt_name, validated_gt_subset_id, validated_gt_attribute_id, default_costs, offset_costs_combined, True))

    # original gt
    original_gt_name = "Original GT"
    org_gt_subset_id = uuid.UUID("100ebd23-c875-4d2c-9844-c83fe631dbaf")
    org_gt_attribute_id : uuid.UUID = None

    ids.append((original_gt_name, org_gt_subset_id, org_gt_attribute_id,default_costs, 0 , True))

    validated_gt_thresholds = [0.5, 0.8]
    min_box_sizes = [0, 25]

    # cascadercnn loss
    comparison_name = "Instance-wise Loss Method"
    comparison_subset_id: uuid.UUID = uuid.UUID("4476db96-1011-496c-a416-24413d1c2fc7")
    comparison_attribute_id: uuid.UUID = uuid.UUID("58a7c441-f914-4ddf-bf5d-3a8db998b61e")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id, default_costs, 0, False))


    # yolox
    comparison_name = "YOLOX"
    comparison_subset_id: uuid.UUID = uuid.UUID("da1db511-f94f-477c-b8ea-1e09b80b8f05")
    #v2 includes metadetect and obj score
    comparison_subset_id: uuid.UUID = uuid.UUID("290c26ea-306f-4bb5-9542-abe8e6837480")
    comparison_attribute_id: uuid.UUID = uuid.UUID("ca555d5e-9ea7-45d7-bc86-01b1a3539812")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id,default_costs, 0, False))
    # meta detect
    comparison_name = "YOLOX+MetaDetect"
    comparison_attribute_id: uuid.UUID = uuid.UUID("6a3dfb0d-3348-43e7-af2e-16030f5e8941")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id, default_costs, 0, False))
    # cleanlab
    comparison_name = "YOLOX+ObjectLab"
    comparison_attribute_id: uuid.UUID = uuid.UUID("fd98e1d8-d2b4-45e3-9218-099dca4e4cec")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id, default_costs, 0, False))

    # cascadercnn
    comparison_name = "Cascade R-CNN"
    comparison_subset_id: uuid.UUID = uuid.UUID("91bd6290-9d28-4487-b96f-c9bf37486d67")
    # v2 includes metadetect and obj score
    comparison_subset_id: uuid.UUID = uuid.UUID("d114436c-91ec-4d8f-b214-c7a501e45bbe")

    comparison_attribute_id: uuid.UUID = uuid.UUID("ca555d5e-9ea7-45d7-bc86-01b1a3539812")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id,default_costs,0,  False))
    # meta detect
    comparison_name = "Cascade R-CNN+MetaDetect"
    comparison_attribute_id: uuid.UUID = uuid.UUID("6a3dfb0d-3348-43e7-af2e-16030f5e8941")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id, default_costs,0, False))
    # cleanlab
    comparison_name = "Cascade R-CNN+ObjectLab"
    comparison_attribute_id: uuid.UUID = uuid.UUID("fd98e1d8-d2b4-45e3-9218-099dca4e4cec")
    ids.append((comparison_name, comparison_subset_id, comparison_attribute_id, default_costs,0, False))



    ## box generation
    box_gen = []
    # Keypoint + BBOx
    comparison_name = "Keypoint-to-box Annotation"
    comparison_subset_id: uuid.UUID = uuid.UUID("f9d26057-4a6c-449a-b474-e99a8c4d8bf4")
    box_gen.append((comparison_name, comparison_subset_id, offset_costs_keypoint_box))

    #  BBox
    comparison_name = "Direct Box Annotation"
    comparison_subset_id: uuid.UUID = uuid.UUID("966059b3-c265-4291-9b71-a525af73ca61")
    box_gen.append((comparison_name, comparison_subset_id, offset_costs_bboxes))

    # combined bboxes
    # non-grouped GT
    comparison_name = "Combined Box Annotation"
    #v2
    comparison_subset_id: uuid.UUID = uuid.UUID("ec08d1db-eae7-46ce-b4e1-f3edbafdb659")
    #v5
    comparison_subset_id: uuid.UUID = uuid.UUID("8a47be9a-b23c-4df9-ab1c-000e01971ce3")
    box_gen.append((comparison_name, comparison_subset_id, offset_costs_combined))


    # LABELING STRATEGIES
    label_strats = []
    # Is correct
    #v3
    comparison_attribute_id: uuid.UUID = uuid.UUID("4987e5b1-a9ff-42e9-a31e-af328df7803d")
    #v5
    comparison_attribute_id: uuid.UUID = uuid.UUID("3c20d055-3da7-41f4-a8f4-5c09eab926fa")
    label_strats.append(("Is pedestrian?",comparison_attribute_id, 4.7339))

    # person + walking?
    #v3
    comparison_attribute_id: uuid.UUID = uuid.UUID("24d46e3d-38cd-4cd7-8def-f9f9267ba9b0")
    #v5
    comparison_attribute_id: uuid.UUID = uuid.UUID("bfb5efdb-1874-4d1d-b963-1eebfa32dac0")
    label_strats.append(("Is human \& stand/walk?", comparison_attribute_id, 7.1286))

    # person + walking? + ambigous ( non-grouped GT)
    #v3
    comparison_attribute_id: uuid.UUID = uuid.UUID("09bcb7c8-5b3f-43ee-b4db-47c24465ffcb")
    #v5
    comparison_attribute_id: uuid.UUID = uuid.UUID("9f09051e-36f2-4f22-ab44-1146cb625984")
    label_strats.append(("Is human \& stand/walk? \& AR", comparison_attribute_id, 9.7318))


    # add all combinations of label and box strategy
    for box in box_gen:
        for label_strat in label_strats:
            ids.append((r'\shortstack[l]{' + box[0] +"+\\\\"+label_strat[0] + "}", box[1], label_strat[1], label_strat[2], box[2], True))


    # person + walking?  + ambiguous + group (validated GT)

    # # sanity check
    # comparison_name = "SANITY_CHECK"
    # comparison_subset_id = uuid.UUID("d112d09e-780c-4b2b-a37a-2331b2e9478d")
    # comparison_attribute_id: uuid.UUID = uuid.UUID("85f54ed9-db72-489f-8c64-63e5ce227c3d")
    # ids.append((comparison_name, comparison_subset_id, comparison_attribute_id))

    # add combined error corretion methods for non -labeling strategies
    # non_labeling_strats = [id for id in ids if not id[4]]
    # for nls in non_labeling_strats:
    #
    #     ids.append((nls[0]+"_corrected",[org_gt_subset_id, nls[1]],nls[2],nls[3],nls[4]))

    ious_to_sweep = [0.1, 0.5]  # 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8, 0.9]
    confidences_to_sweep = [0,
                            0.01, 0.1,
                            # 0.2,
                            0.3,  # 0.4,
                            0.5,  # 0.6,
                            0.7,  # 0.8,
                            0.9, 0.95,   0.99,
                            1.0]

    # load hari client
    config: Config = Config(_env_file=".env")
    hari: HARIClient = HARIClient(config=config)

    # define cache directory
    cache_directory = join(config.data_directory, cache_directory_name)

    medias, media_objects, attributes, attribute_metas = collect_media_and_attributes(
        hari,
        dataset_id,
        cache_directory,
        subset_ids=[],
        additional_fields=["attributes"],
    )

    # generate lookup tables
    ID2attribute_meta = {a.id: a for a in attribute_metas}
    mo_id_to_mo = {mo.id: mo for mo in media_objects} # Media object id to media object


    # calculate bboxes and scores
    bboxes, scores, mo_ids = {},{}, {}
    for name, subset_id, attribute_id, cost_per_sample, offset_costs, is_labeling_strategy in ids:
    # for name, subset_id, attribute_id in zip([comparison_name, validated_gt_name], [comparison_subset_id,validated_gt_subset_id], [comparison_attribute_id,validated_gt_attribute_id]):



        if isinstance(subset_id, list):
            # special case combined original gt and method
            filtered_mo_ids = [mo.id for mo in media_objects if str(subset_id[0]) in mo.subset_ids or str(subset_id[1]) in mo.subset_ids]
            default_score = 1
        else:
            # filter for specified subset id
            filtered_mo_ids = [mo.id for mo in media_objects if str(subset_id) in mo.subset_ids]
            default_score = None
        mo_ids[name] = filtered_mo_ids

        # for mo in media_objects:
        #     print(mo.subset_ids, subset_id)

        print(f"{name} - Found {len(filtered_mo_ids)} media objects (bounding boxes)")



        # get score
        def get_score(attributes):
            if attribute_id is not None:
                for attribute in attributes:
                    if attribute.id == str(attribute_id):
                        return attribute.value
                if default_score is not None:
                    return default_score
                raise ValueError(f"Could not find score for {attribute_id}", attributes)
            else:
                return 1.0


        print(f"{name} - Extract scores ...")

        scores[name] = [
            get_score(mo.attributes)
            for mo_id in filtered_mo_ids
            if (mo := mo_id_to_mo.get(mo_id))
        ]



        def get_bbox(media_object):
            if media_object.reference_data:
                bbox_data = media_object.reference_data
            elif media_object.qm_data:
                bbox_data = media_object.qm_data[0]
                assert bbox_data.type == "bbox2d_center_point_aggregation"
            else:
                raise ValueError

            return [
                bbox_data.x,
                bbox_data.y,
                bbox_data.width,
                bbox_data.height
            ]


        print(f"{name} - Extract bboxes ...")

        bboxes[name] = [
            get_bbox(mo)
            for mo_id in filtered_mo_ids
            if (mo := mo_id_to_mo.get(mo_id))
        ]

    for validated_gt_threshold in validated_gt_thresholds:
        for min_box_size in min_box_sizes:
            print(f"Analyzing for validated gt threshold: {validated_gt_threshold} and min box size: {min_box_size}")

            cache_file = f'all_comparisons_{validated_gt_threshold}_{min_box_size}.pkl'

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    print(f"Loaded cached dictionary from {cache_file}")
                    all_comparisons =  pickle.load(f)
            else:
                all_comparisons = {}

                for comparison_name, subset_id, _, cost_per_sample, offset_costs, is_labeling_strategy in ids:
                    print(f"#### Analyzing {comparison_name}")

                    if isinstance(subset_id, list):
                        is_correction_method = True
                    else:
                        is_correction_method = False


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

                    # get media ids
                    gt_media_ids = [mo_id_to_mo[mo_id].media_id for mo_id in gt_mo_ids]
                    org_gt_media_ids = [mo_id_to_mo[mo_id].media_id for mo_id in org_gt_mo_ids]
                    comparison_media_ids = [mo_id_to_mo[mo_id].media_id for mo_id in comparison_mo_ids]
                    unique_media_ids = list(set(gt_media_ids + comparison_media_ids + org_gt_mo_ids))

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

                                # filter out matched predictions with orignal gt if corrected method
                                # if is_correction_method:
                                #     assert len(matched_org_gt) == len(org_gt_bboxes_img) # all org gt should be matched
                                #     # remove predictions matching with original gt
                                #     non_original_predictions = [i for i in range(len(pred_bboxes_img)) if i not in matched_org_gt_pred]
                                #     non_original_pred_bboxes = [box for idx, box in enumerate(pred_bboxes_img) if idx in non_original_predictions]
                                #     non_original_pred_scores = [score for idx, score in enumerate(pred_scores_img) if idx in non_original_predictions]
                                #     # keep original gt
                                #     pred_bboxes_img = [box for idx, box in enumerate(pred_bboxes_img) if idx in matched_org_gt_pred]
                                #     pred_scores_img = [score for idx, score in enumerate(pred_scores_img) if idx in matched_org_gt_pred]
                                #
                                #
                                #     # calculate iou
                                #     iou_matrix = compute_iou_matrix(org_gt_bboxes_img, non_original_pred_bboxes)
                                #
                                #     for idx, (pred_bbox, pred_score) in enumerate(zip(non_original_pred_bboxes,non_original_pred_scores)):
                                #
                                #         # try to find match with original gt
                                #         max_iou = max(iou_matrix[:, idx]) if len(
                                #             iou_matrix[:, idx]) > 0 else 0  # maximum iou with gt boxes
                                #         if max_iou < 0.5:
                                #             # non original gt and not matching with any -> keep
                                #             pred_bboxes_img.append(pred_bbox)
                                #             pred_scores_img.append(pred_score)
                                #
                                #     # recalculate -> could simplify since it must be the first entries which are matched
                                #     _, _, _, matched_org_gt, matched_org_gt_pred = (
                                #         match_and_count(org_gt_bboxes_img, None,
                                #                         pred_bboxes_img, pred_scores_img,
                                #                         0.5))  # use realistic iou due to matching

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
                                # label_errors_raw, c = detect_label_errors(org_gt_bboxes_img, pred_bboxes_img,
                                #                                           matched_gt_pred, 0.5, cost_per_sample)
                                # # determine percentage of label errors
                                # labels, counts = np.unique(label_errors_raw, return_counts=True)
                                # label_to_count = dict(zip(labels, counts))
                                # label_error = label_to_count.get('overlooked_pedestrian', 0) + label_to_count.get(
                                #     'misfitting_box', 0)
                                # label_okay = label_to_count.get('no_tn', 0) + label_to_count.get('no_tp', 0)


                                # v2 with greedy gt matching
                                label_errors_raw, c = detect_label_errorsv2(len(pred_bboxes_img),pred_scores_img, matched_gt, matched_gt_pred,
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




                                    # for idx, moid in enumerate(pred_moid_img):
                                    #     if moid not in reliable_labels:
                                    #         # print("no id:", moid)
                                    #         pass
                                    #     else:
                                    #         reliable_label = reliable_labels[moid]
                                    #         if not reliable_label['label_error'] and moid in label_error_fn_moids:
                                    #             print("MISSING:", moid)
                                    #             print('score ', pred_scores_img[idx], ' bbox ', pred_bboxes_img[idx], ' mached with validated GT ', idx in matched_gt_pred , ' matched with orignal gt ', idx in matched_org_gt_pred, " error " , label_errors_raw[idx])
                                    #
                                    #             print(reliable_label)
                                    #     if moid in label_error_moid_list:
                                    #         print("DUPLICATE:", moid)
                                    # label_error_moid_list.extend(label_error_fn_moids)
                                    # print(len(label_error_moid_list))







                                # print(label_to_count)

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

                    # Plot
                    # plt.figure(figsize=(10, 7))
                    # cmap = plt.get_cmap("viridis")  # color map for different IoUs
                    # num_colors = len(ious_to_sweep)
                    #
                    # # show all metrics in comparison to one main metric
                    # names_metrics = ["Total Costs [ct]",
                    #                  "False Positive Rate (FPR)",
                    #                  "False Negative Rate (FNR)",
                    #                  "Mean Absolute Error (MAE)",
                    #                  "Matched with Validated GT [%]",
                    #                  "Label Error [#]",
                    #                  "Label Okay [#]",
                    #                  "Overlooked FP in Org. GT  [#]",
                    #                  "Label Error (Missing BBox) [#]",
                    #                  "Total GT", "Total Pred"]
                    # primary_index = 0
                    #
                    # # Define markers (must be at least as many as confidences_to_sweep)
                    # marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']
                    #
                    # for idx_metric, name in enumerate(names_metrics):
                    #     if idx_metric == primary_index:
                    #         continue # no self comparison
                    #
                    #     for idx, (iou_thresh, metric_lists) in enumerate(all_curves.items()):
                    #         color = cmap(idx / (num_colors - 1))
                    #         plt.plot(metric_lists[primary_index], metric_lists[idx_metric], marker='o', color=color, label=f"IoU={iou_thresh:.2f}")
                    #
                    #         # Highlight lowest confidence with a cross
                    #         plt.plot(metric_lists[primary_index][0], metric_lists[idx_metric][0],
                    #                  color='black', marker='x', markersize=9, linestyle='None',
                    #                  label=None)
                    #
                    #     plt.xlabel(names_metrics[primary_index])
                    #     plt.ylabel(names_metrics[idx_metric])
                    #     plt.title(comparison_name)
                    #     plt.legend(title="IoU Threshold")
                    #     plt.grid(True)
                    #     plt.tight_layout()
                    #     plt.show()

                    # store all curves for per method comparison
                    all_comparisons[comparison_name] = all_curves

                    # CONVERT TO COCO
                    # category_id = 1 # pedestrian
                    # gt_annotations = {
                    #     "images": [{"id": media_id } for media_id in unique_media_ids],
                    #     "annotations": [],
                    #     "categories": [{"id": 0, "name": "background"},{"id": category_id, "name": "pedestrian"}],
                    #     "info": {"description": "Reliable dataset"},
                    #     "licenses": [],
                    # }
                    #
                    # for idx, box in enumerate(gt_bboxes):
                    #     gt_annotations["annotations"].append({
                    #         "id": idx,
                    #         "image_id": gt_media_ids[idx],
                    #         "category_id": category_id,
                    #         "bbox": box,
                    #         "area": box[2] * box[3],
                    #         "iscrowd": 0
                    #     })
                    #
                    # # Build COCO-style prediction annotations
                    # pred_annotations = []
                    # for idx, (box, score) in enumerate(zip(comparison_bboxes, comparison_scores)):
                    #     pred_annotations.append({
                    #         "image_id": comparison_media_ids[idx],
                    #         "category_id": category_id,
                    #         "bbox": box,
                    #         "score": score
                    #     })
                    #
                    #
                    #
                    # # COCO evaluation
                    # coco_gt = COCO()
                    # coco_gt.dataset = gt_annotations
                    # coco_gt.createIndex()
                    #
                    # coco_dt = coco_gt.loadRes(pred_annotations)
                    #
                    # # Run evaluation
                    # coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
                    # coco_eval.evaluate()
                    # coco_eval.accumulate()
                    # coco_eval.summarize()

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

            primary_index = 0

            # iterate over confidence treshold
            iou_threshold = 0.1
            # iterate over iou threhsolds
            # iou_threshold = None
            confidence_threshold_index = 0

            names_metrics = ["Total Time [h]",
                             "False Positive Rate (FPR)",
                             "False Negative Rate (FNR)",
                             "Mean Absolute Error (MAE)",
                             "Matched with Validated GT [\%]",
                             "Found Label Errors - FN in Org. GT[\#]",
                             "Label Okay [\#]",
                             "Overlooked Label Errors - FP in Org. GT [\#]",
                             "Introduced Label Errors [\#]",
                             "Total GT", "Total Pred"]

            # plot_params = set_plot_layout(path_to_latex='/usr/local/texlive/2025/bin/universal-darwin')

            for idx_metric, name in enumerate(names_metrics):
                print("###" + name + "###")
                if idx_metric == primary_index:
                    continue  # no self comparison

                lines = []
                plt.figure(figsize=(12,8))

                for idx, (comparison_name, all_curves) in enumerate(all_comparisons.items()):

                    # rules to filter some plots out
                    # if "FNR" in name or "MAE" in name:
                    #     if "YOLOX" in comparison_name or "CascadeRCNN" in comparison_name:
                    #         continue
                    #
                    # if "Label" in name or "Overlooked" in name:
                    #     if "Annotation" in comparison_name:
                    #         continue




                    color = cmap[idx] # / (num_colors - 1))
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
                    line = plt.plot([(x/100)/4.5 for x in metric_lists[primary_index]], metric_lists[idx_metric], marker='o', color=color,
                             label=comparison_name)
                    lines.append(line)

                    print(f"{comparison_name}@{confidences_to_sweep[confidence_threshold_index]:0.02f}@{iou_threshold:0.02f}:"
                          f"{metric_lists[idx_metric][confidence_threshold_index]:0.02f}")

                    # Highlight lowest confidence with a cross
                    plt.plot((metric_lists[primary_index][0]/ 100) / 4.5, metric_lists[idx_metric][0],
                             color='black', marker='x', markersize=9, linestyle='None',
                             label=None)

                plt.xlabel(names_metrics[primary_index])
                plt.ylabel(names_metrics[idx_metric])
                # plt.legend(title="Method")
                plt.grid(True)
                plt.tight_layout()
                # plt.show()
                plt.savefig(f"plots/{validated_gt_threshold}_{min_box_size}_{name.split('[')[0]}.png", bbox_inches='tight', dpi=300)

            # Create a new figure just for the legend
            fig_legend = plt.figure(figsize=(3, 1))  # adjust size as needed
            fig_legend.legend(
                handles=[line[0] for line in lines],
                labels=[name for name in all_comparisons.keys()],
                loc='center',
                frameon=False
            )

            fig_legend.savefig("plots/legend.png", bbox_inches='tight', dpi=300)
