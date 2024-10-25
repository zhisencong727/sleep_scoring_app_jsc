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

    trainingFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring_app_jsc/app_src/trainingFileList.txt")
    testingFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring_app_jsc/app_src/testingFileList.txt")
    goldStandardFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring_app_jsc/app_src/goldStandardFileList.txt")

    dict = {}

    confusionMatrix = [[0,0,0],[0,0,0],[0,0,0]]

    for each in goldStandardFileList:
        mat_file = os.path.join(data_path, each)
        mat = loadmat(mat_file)
        mat, output_path = run_inference(mat, model_choice, postprocess=False)

        

        #print(each)
        get_confusionMatrix(loadmat(mat_file),mat,confusionMatrix)

        print(each)
        dict[each] = f1_score_evaluation(loadmat(mat_file),mat)
        
        for row in confusionMatrix:
            print(row)
    
    for each in dict.keys():
        print(each,end=":")
        print(dict[each])