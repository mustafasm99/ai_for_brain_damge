import cv2                                                  # computer vesion 
import numpy as np                                          # number array manage use to do all mathmatics on the matrses 
from sklearn.base import BaseEstimator, TransformerMixin    # two calsses to inhertanse from tah will lead the modle and the image adjsmint 
import joblib                                               # to read the the model from file that we tranin in the main file 
import sys
import os 

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# the main libs for the loaded after makeing the exe fiel 
from sklearn.model_selection        import train_test_split
from sklearn.svm                    import SVC
from sklearn.pipeline               import Pipeline
from sklearn.metrics                import accuracy_score
from sklearn.decomposition          import PCA
import pandas as pd 


def resource_path(path):
    bpath   = os.path.abspath(".")
    print(bpath)
    return os.path.join(bpath , path)

# Define the ImageLoader class here
class ImageLoader(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Read and resize images
        print(X)
        images = [cv2.imread(path) for path in X.flatten()]
        resized_images = [cv2.resize(img, (128, 128)) for img in images]

        # Stack images into a 3D array
        stacked_images = np.array(resized_images)

        # Reshape the 3D array to 2D for PCA
        reshaped_images = stacked_images.reshape(stacked_images.shape[0], -1)

        return reshaped_images

# Load the saved 

# Load the saved model
loaded_pipeline = joblib.load(resource_path("model.joblib"))

# loaded_pipeline = joblib.load(resource_path('model.joblib'))


#  ! dawn the class for the gui 

class ImageApp:
    def __init__(self , root):
        self.root = root
        self.root.geometry("300x550")
        self.root.title = "Pitiuitary Detections"
        
        self.canvas = tk.Canvas(root , width=450 , height=450)
        self.canvas.pack()
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack()
        
        self.load_buttons = tk.Button(root , text="load image" , command=self.load_images)
        self.load_buttons.pack()
        
        
        self.predict_button = tk.Button(root , text="predict" , command=self.predict_imae)
        self.predict_button.pack()
        
    def load_images(self):
        file_path   = filedialog.askopenfilename(filetypes=[("Image files" , "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.image = Image.open(file_path)
            self.image.thumbnail((300,300))
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0,0,anchor="nw" ,image=self.tk_image)
            
            
            self.file_path = file_path
            self.result_label.config(text="")
    def predict_imae(self):
        if hasattr(self , 'file_path'): # check if the image was loaded right with checking the if the file path was set 
            prediction      = loaded_pipeline.predict(np.array([self.file_path]).reshape(-1, 1))
            
            if prediction[0] == 1:  # Assuming 1 corresponds to the affected class
                self.result_label.config(text="The image shows pituitary.")
                original_size = (128,128)
                des_size      = (300,300)
                
                x,y,w,h     = 50,50,30,30
                scale_x     = des_size[0]/original_size[0]
                scale_y     = des_size[1]/original_size[1]
                
                x,y,w,h     = int(x*scale_x) , int(y*scale_y) , int(w*scale_x) , int(h*scale_y)
    
                self.canvas.create_rectangle(x, y, x + w, y + h, outline="red", width=2)
                print("done create image ")
            else:
                self.result_label.config(text="The image does not show pituitary.")
        else:
            self.result_label.config(text="Please load an image first.")
                
        

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ImageApp(root)
        root.mainloop()
    except Exception as error:
        print(error)
        input("Press Enter to exit...")


# Use the loaded model for prediction on new image(s)
# new_image_path = 'data/pituitary/Tr-pi_0010.jpg'
# prediction = loaded_pipeline.predict(np.array([new_image_path]).reshape(-1, 1))


# print(f'Prediction for {new_image_path}: {prediction[0]}')

# image = cv2.resize( cv2.imread(new_image_path) , (128,128))
# # fimage = image.reshape(1,-1)

# if prediction[0] == 1:
#     x, y, w, h = 50, 50, 30, 30  # Example coordinates and dimensions
#     # Draw the rectangle on the image
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()