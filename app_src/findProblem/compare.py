from scipy.io import loadmat
import numpy as np
import math

groundtruth = loadmat('findProblem/groundtruth_data/nor_484_yfp.mat')
predicted = loadmat('findProblem/prediction_data/nor_484_yfp_sdreamer_3class.mat')

gt_score = groundtruth['sleep_scores'].flatten()
pred_score = predicted['pred_labels'].flatten()

gt_score = gt_score[0:len(pred_score)]
match_count = np.sum(gt_score==pred_score)
total = len(gt_score)
difference_count = total-match_count

wrongArr = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]

sleep_as_wake_count = 0
for i in range(len(pred_score)):
    if math.isnan(gt_score[i]):
        #print("nan i is:",i)
        #print("predicted for the nan-i is:",pred_score[i])
        print("NaN is detected in this file")
        continue
    if pred_score[i] != gt_score[i]:
        wrongArr[int(gt_score[i])][pred_score[i]] += 1

print(wrongArr)
        


print(f"Number of matching elements: {match_count}")
print(f"Number of differing elements: {difference_count}")
