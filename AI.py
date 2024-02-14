import pandas as pd                                     # read the csv file 
from sklearn.model_selection import train_test_split    # split the data to train and test


data = pd.read_csv("labels.csv")                        
train_df, test_df = train_test_split(data, test_size=0.2, random_state=50)

from keras.utils import image_dataset_from_directory

datagen     = image_dataset_from_directory("data" , label_mode="int" , labels="inferred" )
print(datagen)