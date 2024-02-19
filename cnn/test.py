from keras.preprocessing import image
import numpy as np 
from keras import models

model = models.load_model("CNN_model.h5")
# Load and preprocess the test image
img_path = 'data\pituitary\Tr-pi_0010.jpg'
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
prediction = model.predict(img_array)
print('Probability of having cancer:', prediction[0][0])