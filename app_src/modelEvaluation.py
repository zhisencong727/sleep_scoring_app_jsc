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
    
    gt_score = groundtruth['sleep_scores'].flatten()
    pred_score = predicted['pred_labels'].flatten()

    gt_score = gt_score[0:len(pred_score)]

    
    for i in range(len(gt_score)):
        confusionMatrix[int(gt_score[i])][int(pred_score[i])] += 1
    





#f1_score_evaluation(loadmat("app_src/groundtruth_data/sal_486.mat"),loadmat("app_src/prediction_data/nor_484_yfp_sdreamer_3class.mat"))


