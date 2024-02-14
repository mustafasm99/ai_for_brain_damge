from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("labels.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the image preprocessing pipeline
image_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50))  # You can adjust the number of components based on your data
])

X_train = image_pipeline.fit_transform(train_df['image_path'].values.reshape(-1, 1))

# Transform the test data using the same pipeline
X_test = image_pipeline.transform(test_df['image_path'].values.reshape(-1, 1))

y_train = train_df['label']
y_test = test_df['label']

svm_classifier = SVC()
svm_classifier.fit(X_train , y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)