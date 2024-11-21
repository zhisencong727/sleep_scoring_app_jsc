# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 2024

@author: zcong and yzhao
"""

import os
from scipy.io import loadmat, savemat

import run_inference_ne as run_inference_ne
from app_src.postprocessing import postprocess_pred_labels
from app_src.modelEvaluation import f1_score_evaluation
from app_src.modelEvaluation import get_confusionMatrix
from app_src.modelEvaluation import scoreCalculationFromCM,find_periods,compute_iou,overall_iou


MODEL_PATH = "app_src/models/sdreamer/ne_model_result/"


def run_inference(
    mat, model_choice="sdreamer", num_class=3, postprocess=False, output_path=None
):
    # num_class = 3
    predictions, confidence = run_inference_ne.infer(mat, MODEL_PATH)
    mat["pred_labels"] = predictions
    mat["confidence"] = confidence
    # mat["num_class"] = 3
    if postprocess:
        print("POST!")
        predictions = postprocess_pred_labels(mat)
        mat["pred_labels"] = predictions

    if output_path is not None:
        output_path = (
            os.path.splitext(output_path)[0] + f"_sdreamer_{num_class}class.mat"
        )
        savemat(output_path, mat)
    return mat, output_path

def getListFromFile(file):
    ret = []
    with open(file,"r") as f:
        for line in f:
            ret.append(line.strip())
    return ret

if __name__ == "__main__":


    model_choice = "sdreamer"
    data_path = "/Users/jsc727/Documents/sdreamer_train_jsc/groundtruth_data"

    trainingFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring_app_jsc_Github/app_src/trainingFileList.txt")
    testingFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring_app_jsc_Github/app_src/testingFileList.txt")
    goldStandardFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring_app_jsc_Github/app_src/goldStandardFileList.txt")

    dict = {}

    confusionMatrix = [[0,0,0],[0,0,0],[0,0,0]]

    for each in testingFileList:
        mat_file = os.path.join(data_path, each)
        mat = loadmat(mat_file)
        mat, output_path = run_inference(mat, model_choice, postprocess=False)

        gt = mat['sleep_scores'].flatten()
        pred = mat['pred_labels'].flatten()
     
        gt_rem_period = find_periods(gt,2)
        print("Here is the IoU List:")
        iou_list = compute_iou(gt,pred,gt_rem_period,2)
       
        print(each,end=": ")
        overall_iou(gt,pred,2)

        individual_cm = get_confusionMatrix(loadmat(mat_file),mat,confusionMatrix)
        scoreCalculationFromCM(individual_cm)

        
        dict[each] = f1_score_evaluation(loadmat(mat_file),mat)
        
        
    
    for each in dict.keys():
        print(each,end=":")
        print(dict[each])
    
    for row in confusionMatrix:
        print(row)
    
    scoreCalculationFromCM(confusionMatrix)