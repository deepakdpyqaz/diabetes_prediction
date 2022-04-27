from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv("processed_data/diabetes.csv")
X = data.drop("Outcome",axis=1)
X = X.drop("Unnamed: 0",axis=1)
X_train,X_test = train_test_split(X,test_size=0.2)
model = keras.models.load_model("serving_model/dibetes_prediction_pipeline/1650624722")
model.summary()
x = np.array([0])
print(x)
print(model.predict([x,x,x,x,x,x,x,x]))