import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets

iris = datasets.load_iris()
clases = iris.target_names

# Title of the app
st.title('Iris Flower Classification')

# Description of the task
st.markdown("""
### Task: Building and Deploying Machine Learning Models for the Iris Dataset

**Description:**
For this task, I configured and trained four different machine learning models to classify flowers from the Iris dataset that is available on the sci-kitlearn library into three categories: Iris Setosa, Iris Versicolour, and Iris Virginica. I then deployed these models using Flask to provide a web interface for making predictions. Additionally, I created a client to interact with the server and test the models. The entire project was managed using Poetry for dependency management and Visual Studio Code as the development environment.

**How I Did It:**

1. **Data Preparation:**
   - **Loading Data:** Loaded the Iris dataset, which contains measurements of sepal length, sepal width, petal length, and petal width for three Iris species.
   - **Feature Selection:** Selected petal length and petal width as the features for the classification models.
   - **Data Splitting:** Split the dataset into training (70%) and testing (30%) sets.
   - **Normalization:** Normalized the data for most models except for decision trees using standard scaling, ensuring the mean is 0 and the standard deviation is 1, based on the training set. Applied the same transformation to the test set.

2. **Model Training:**
   I implemented and trained the following models using Python:
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**
   - **Decision Trees**
   - **k-Nearest Neighbors (KNN)**
   
   For each model:
   - **Configured the Model:** Set up the model parameters.
   - **Trained the Model:** Trained the model using the training data.
   - **Evaluated the Model:** Tested the performance making 3 different curl requests expecting 3 different flower prediction on each model.

3. **Serialization:**
   - Serialized the trained models using `pickle` to save them for later use.

4. **Web Deployment with Flask:**
   - Created a Flask web application to serve the models.
   - Developed endpoints for each model to accept input data and return predictions.

5. **Client Interaction:**
   - Implemented a jupyter notebook to send HTTP requests to the Flask server.
   - Sent three requests to each model endpoint and displayed the responses on the client notebook.

6. **Project Management and Deployment:**
   - **Environment Setup:** Used Poetry to manage dependencies and configure the project environment.
   - **Development:** Used Visual Studio Code for development.
   - **Version Control:** Pushed the project to a public GitHub repository for version control and sharing.
   - **Programing languages:** Written in python using scripts and jupyter notebooks
   - **Frameworks and libraries:** Used Flask to handle HTTP requests & pickle for object serialization

**Skills acquired/improved on:** 
                 
""")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.image('imgs/flask.png', caption='Flask', width=50)

with col2:
    st.image('imgs/github.png', caption='github', width=50)

with col3:
    st.image('imgs/jupyter.png', caption='jupyter', width=50)

with col4:
    st.image('imgs/poetry.png', caption='Poetry', width=50)

with col5:
    st.image('imgs/python.png', caption='python', width=50)

with col6:
    st.image('imgs/vscode.png', caption='vscode', width=50)

st.write("""**Working Demo:**
[Course task nº3](https://github.com/cristobalcanomorey/PIA_tasca_3_Cristobal_Cano)
""")

def predict_single(flor, model):
    # x = dv.transform([flor])
    y_pred = model.predict(flor)
    return y_pred 

def get_prediction(flor, modelo):
    prediction = predict_single(flor, modelo)
    return 'Iris ' + clases[int(prediction[0])]


# Load the models and scalers
with open('models/flor-lr.pck', 'rb') as f:
    lr_model, lr_scaler = pickle.load(f)
with open('models/flor-svm.pck', 'rb') as f:
    svm_model, svm_scaler = pickle.load(f)
with open('models/flor-tree_model.pck', 'rb') as f:
    dt_model = pickle.load(f)
with open('models/flor-knn.pck', 'rb') as f:
    knn_model, knn_scaler = pickle.load(f)

# Sidebar for user input
st.sidebar.header('Input Features')
def user_input_features():
    petal_length = st.sidebar.slider('Petal Length', 0.0, 7.0, 3.5)
    petal_width = st.sidebar.slider('Petal Width', 0.0, 2.5, 1.0)
    data = {'petal_length': petal_length, 'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
st.subheader('Working demo')
st.write("""Use the sidebar sliders to thest the classification models I trained for the course task in real time.
         See their outputs here:
""")
models = {
    'Logistic Regression': (lr_model, lr_scaler),
    'Support Vector Machine': (svm_model, svm_scaler),
    'Decision Tree': (dt_model, False),
    'K-Nearest Neighbors': (knn_model, knn_scaler)
}

for model_name, tuple in models.items():
    model, scaler = tuple
    if scaler:
        # Normalize the input features
        input_features = scaler.transform(input_df)
    else:
        input_features = input_df
    
    prediction = get_prediction(input_features,model)
    #prediction = model.predict(input_features)
    
    st.write(f"### {model_name}")
    st.write(f"Prediction: {prediction}")
    
    if hasattr(model, "predict_proba"):
        prediction_proba = model.predict_proba(input_features)
        st.write(f"Prediction Probability: {prediction_proba[0]}")


# Display the input features
st.subheader('Input Features')
st.write(input_df)
