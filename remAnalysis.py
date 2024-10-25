from app_src.inference import getListFromFile
import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = "/Users/jsc727/Documents/sleep_scoring-main/app_src/groundtruth_data"

allFileList = getListFromFile("/Users/jsc727/Documents/sleep_scoring-main/app_src/allFileList.txt")

remDurationList = []

for each in allFileList:
    mat_file_name = os.path.join(data_path, each)
    groundtruth = loadmat(mat_file_name)

    sleep_scores = groundtruth['sleep_scores'].flatten()
    start = 0
    end = 0
    startBool = False
    i = 0
    while i < len(sleep_scores):
        if sleep_scores[i] == 2.0 and startBool == False:
            start = i
            startBool = True
        elif sleep_scores[i] != 2.0 and startBool == True:
            end = i
            #print("start is",start)
            #print("end is",end)
            
            startBool = False
            remDurationList.append(end-start)
            startBool = False
        i += 1


        
print(remDurationList)

prctArr = pd.Series(remDurationList)

print(prctArr.describe())

plt.figure(figsize=(8, 6))
plt.hist(np.array(remDurationList), bins='auto', color='skyblue', edgecolor='black')
plt.title('Histogram of Duration')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()