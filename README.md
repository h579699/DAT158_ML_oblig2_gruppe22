# DAT158 ML oblig2 group 22 - Prediction of house prices

App-link: https://dat158-group22-oblig2.herokuapp.com/

Project Organization
--------
    ├── images              <- Images
    │   
    ├── model              <- trained model stored as .pkl
    ├── notebooks          <- Jupyter notebooks 
    │    
    │   
    ├── app.py             <- Exported streamlit application
    ├── Procfile           <- Code that provides startup instructions to the web server
    ├── requirements.txt   <- names of the python packages required to execute the application
    ├── setup.sh           <- Bash script used for creating the necessary environment for our streamlit app to run on the cloud.
    │
    └── README.md          <- The top-level README for developers using this project
--------


Project Overview
--------
DAT158 Machine Learning Oblig 2, Group 22: Oneal Didrik Ferkingstad Lane and Jaran Jonasson

## Introduction
The purpose of this assignment is to work through a machine learning project and end up with a deployed model. First we study our problem, start exploring the data, and create a machine learning model. After this we put or final model into production by deploying out model. 

## Machine learning model
The House-prices.ipynb notebook contains the code, explainations and considerations for creating our model. Our goal was to create a model for predicting the sale price of houses based on 79 describing features. This problem is based on the kaggle-competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview  

In our work we have used the PyCaret library for data preprocessing, training and comparing models, hyperparameter tuning, analyzation and exporting. The notebook produces a final model called sale_price_pred.pkl, which is a blender consisting of the four best models from PyCaret based on their RMSLE (Root Mean Squared Log Error).

### Considerations
* The PyCaret hyperparameter tuning is based on a randomized grid-search and doesn't necessarily produce a better result than the default parameters. The group was satisfied with the result from the model while using default parameters, and did not get a better result from their hyperparameter tuning. The models used in the blender (final model) therefore uses default parameters. It was also a requirement that it should be easy to reproduce the results for others/run the code, and extensive hyperparameter tuning can take a lot of time. 
* Since a lot of the data preprocessing is done by the PyCaret setup method, the group's data preparations has involved experimenting with different setup configurations. Some of them are listed together with the final setup, in the notebook. 

Check out the notebook House-prices.ipynb for further details.

### Further work
* In further work with the project, the group would experiment with trying to get better hyperparameters from the tuning, and maybe consider making a blender consisting of some tuned models and some models using defaults. 
* The group would've also liked to explore new combinations of features, but found it difficult to implement this in the PyCaret setup method.  

## Model deployment
The House-prices-streamlit-app.ipynb contains the code for creating a streamlit-app for deploying our model. The code is inspired by the deployment tutorials from the course's GitHub repository (https://github.com/skaliy/dat158-ml-course21/tree/main/deployment_tutorials). 

The code for the streamlit-application can be found in the app.py file. The application supports both Batch (by uploading a csv-file) and Online predictions. 

The application is also deployed online on Heroku: https://dat158-group22-oblig2.herokuapp.com/

### Considerations
* Our final model is trained on 67 features. To get the best possible result from the online prediction, the group has chosen to include 66 of these as user inputs (exluding sale price) in the app. The group tried experiementing with fewer features and using default values for the missing features, but the results were not as good as we wanted. 
* The group has spent a lot of effort on making sure that the user inputs are user friendly. This includes making the displayed values of features more understandable. Therefore we have created a lot of display values (shown to user) that correspond with their actual values when sending the user input to the model.  

### Further work
* In further work with the application, the group would consider trying to reduce the amount of user inputs, and finding better default values for missing values. 
