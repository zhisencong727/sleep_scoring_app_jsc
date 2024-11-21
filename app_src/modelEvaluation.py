from scipy.io import loadmat
import numpy as np
from sklearn.metrics import f1_score


def f1_score_evaluation(groundtruth,predicted):

    gt_score = groundtruth['sleep_scores'].flatten()
    pred_score = predicted['pred_labels'].flatten()

    if len(gt_score) < len(pred_score):
        pred_score = pred_score[0:len(gt_score)]
    else:
        gt_score = gt_score[0:len(pred_score)]

    gt_score = np.array(gt_score)
    pred_score = np.array(pred_score)

    f1 = f1_score(gt_score,pred_score,average="weighted")

    print(round(f1,3))
    return round(f1,3)


def get_confusionMatrix(groundtruth,predicted,confusionMatrix):

    individualConfusionMatrix = [[0,0,0],[0,0,0],[0,0,0]]
    
    gt_score = groundtruth['sleep_scores'].flatten()
    pred_score = predicted['pred_labels'].flatten()

    if len(gt_score) < len(pred_score):
        pred_score = pred_score[0:len(gt_score)]
    else:
        gt_score = gt_score[0:len(pred_score)]

    
    for i in range(len(gt_score)):
        confusionMatrix[int(gt_score[i])][int(pred_score[i])] += 1
        individualConfusionMatrix[int(gt_score[i])][int(pred_score[i])] += 1
    print("individualConfusionMatrix:")
    for row in individualConfusionMatrix:
        print(row)
    return individualConfusionMatrix


def find_periods(labels, target_label):
    labels = np.array(labels)
    is_target = (labels == target_label).astype(int)
    diff = np.diff(is_target, prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Handle case where the last label is the target label
    if is_target[-1] == 1:
        ends = np.append(ends, len(labels))

    periods = list(zip(starts+1, ends))
    print(periods)
    return periods


def compute_iou(gt_labels, pred_labels, gt_periods, target_label):
    """
    Computes the IoU for each period in gt_periods.

    Args:
        gt_labels (array-like): Ground truth labels.
        pred_labels (array-like): Predicted labels.
        gt_periods (list of tuples): List of (start_idx, end_idx) periods from ground truth.
        target_label (int): The label for REM sleep.

    Returns:
        iou_list (list): List of IoU values for each period.
    """
    iou_list = []
    len_gt = len(gt_labels)
    len_pred = len(pred_labels)
    for start, end in gt_periods:
        # Ensure indices are within the bounds of both arrays
        start_gt = max(0, min(start, len_gt))
        end_gt = max(0, min(end, len_gt))
        start_pred = max(0, min(start, len_pred))
        end_pred = max(0, min(end, len_pred))
        
        # Slice the ground truth and predicted labels
        gt_slice = gt_labels[start_gt:end_gt]
        pred_slice = pred_labels[start_pred:end_pred]
        
        # Adjust slices to have the same length
        min_len = min(len(gt_slice), len(pred_slice))
        if min_len == 0:
            # Skip if there's no overlap
            iou_list.append(0)
            continue
        gt_mask = (gt_slice[:min_len] == target_label)
        pred_mask = (pred_slice[:min_len] == target_label)
        
        # Compute IoU
        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)
        iou = intersection / union if union != 0 else 0
        iou_list.append(round(iou,3))
    print(iou_list)
    return iou_list


def overall_iou(gt_score,pred_score,target):
    min_len = min(len(gt_score), len(pred_score))
    gt_score = gt_score[:min_len]
    pred_score = pred_score[:min_len]

    target_label = 2  # REM sleep label

    # Create boolean masks where labels are equal to the target label
    gt_mask = (gt_score == target_label)
    pred_mask = (pred_score == target_label)

    # Compute intersection and union
    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)

    # Compute overall IoU
    iou = intersection / union if union != 0 else 0

    print(f"Overall IoU for label {target_label}: {round(iou,3)}")



def scoreCalculationFromCM(confusion_matrix):

    num_classes = len(confusion_matrix)

    class_labels = [f'Class_{i}' for i in range(num_classes)]

    metrics = {}
    for i in range(num_classes):
        TP = confusion_matrix[i][i]
        FP = sum(confusion_matrix[j][i] for j in range(num_classes)) - TP
        FN = sum(confusion_matrix[i][j] for j in range(num_classes)) - TP

        precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_i = (2 * precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0

        metrics[class_labels[i]] = {
            'Precision': round(precision_i,3),
            'Recall': round(recall_i,3),
            'F1 Score': round(f1_i,3)
        }

    print(metrics)


#f1_score_evaluation(loadmat("app_src/groundtruth_data/sal_486.mat"),loadmat("app_src/prediction_data/nor_484_yfp_sdreamer_3class.mat"))


