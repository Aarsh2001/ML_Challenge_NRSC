import cv2
import pandas as pd 
import numpy as np
import os
from glob import glob


PATH = './Data_ML-20210718T153802Z-001/Data_ML/target_feature_AOI/'
EXT = "*.csv"
all_csv_files = []
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        all_csv_files.append(file)


for path in all_csv_files:
    img = np.zeros((750,750))
    csv = pd.read_csv(path)
    filename= path.split('/')[4].split('.')[0]
    filepath = os.path.join('/Data_ML-20210718T153802Z-001/Data_ML/target',filename+'.jpg')
# img = cv2.imread('./Data_ML-20210718T153802Z-001/Data_ML/image_chips/1J01.jpg')
# print(img.shape)
    for i in range(csv.shape[0]):
        pro = (csv['WKT'][i].split('MULTIPOLYGON')[1].split('(')[3].split(')')[0].split(','))
        pro = str(pro).replace('-',',').replace('[','').replace(']','').replace('\'','')
        pro = pro.split(',')
        pro = list( map(float, pro) )
        pro= np.array(pro,np.int32).reshape((-1,1,2))
        img = cv2.polylines(img, [pro], 
                        isClosed=True, color = (235,0,0), thickness=2)
    cv2.imwrite(filepath, img)


