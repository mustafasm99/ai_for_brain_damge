import cv2
import numpy                        as np
from sklearn.base                   import BaseEstimator, TransformerMixin
from sklearn.model_selection        import train_test_split
from sklearn.svm                    import SVC
from sklearn.pipeline               import Pipeline
from sklearn.metrics                import accuracy_score
from sklearn.decomposition          import PCA
import pandas as pd 
import joblib

df = pd.read_csv('labels.csv')

class ImageLoader(BaseEstimator , TransformerMixin):
    def __init__(self) -> None:
        pass
    
    def fit(self , X ,Y=None):
        return self
    
    def transform(self, X):
        # Read and resize images
        images = [cv2.imread(path) for path in X.flatten()]
        resized_images = [cv2.resize(img, (128, 128)) for img in images]

        # Stack images into a 3D array
        stacked_images = np.array(resized_images)

        # Reshape the 3D array to 2D for PCA
        reshaped_images = stacked_images.reshape(stacked_images.shape[0], -1)

        return reshaped_images
    
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

image_pipline = Pipeline([
    ('loader',ImageLoader()),
    ('flatten' , PCA(n_components=50)),
    ('classifier' ,SVC())
])

image_pipline.fit(train_df['image_path'].values.reshape(-1,1), train_df['label'])

joblib.dump(image_pipline , 'model.h5')

y_pred = image_pipline.predict(test_df['image_path'].values.reshape(-1, 1))

accuracy = accuracy_score(test_df['label'], y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
