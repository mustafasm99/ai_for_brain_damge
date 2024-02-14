"""

these block or file is for process the data and make it ready to the model 
these will labels all the data in excel file so we can read the excel and found out all the images 
then send them to sktlearn to create two array that handel the spliting for training and testing 

"""

import pandas as pd 
import os 

#  create the dectionry for the data set
data    = { 
    "image_path":[],
    "label"     :[]
}

for label in ['notumor' , 'pituitary']:
    folder_path     = os.path.join('data' , label)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            data['image_path'].append(os.path.join(folder_path,filename))
            data['label'].append(1 if label == 'pituitary' else 0)
            
df = pd.DataFrame(data)
df.to_csv('labels.csv' , index = False)
