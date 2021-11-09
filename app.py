#!/usr/bin/env python
# coding: utf-8

# In[8]:


from pycaret.regression import load_model, predict_model 
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os


# In[9]:


class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('model/sale_price_pred') 
        self.save_fn = 'predictions.csv'     
        
    def predict(self, input_data): 
        return predict_model(self.model, data=input_data)
    
    def store_prediction(self, output_df): 
        if os.path.exists(self.save_fn):
            save_df = pd.read_csv(self.save_fn)
            save_df = save_df.append(output_df, ignore_index=True)
            save_df.to_csv(self.save_fn, index=False)
            
        else: 
            output_df.to_csv(self.save_fn, index=False)  
            
    def preprocess(self, data):
   # Code from Flask-tutorial
   # Makes sure all features are defined going into the model
    
        null = None

        feature_values = {
            'Id':null, 'MSSubClass': null, 'MSZoning': null, 'LotFrontage':null, 'LotArea':null, 'Street': null,
       'Alley': null, 'LotShape': null, 'LandContour':null, 'Utilities':null, 'LotConfig':null,
       'LandSlope':null, 'Neighborhood':null, 'Condition1':null, 'Condition2':null, 'BldgType':null,
       'HouseStyle':null, 'OverallQual':null, 'OverallCond':null, 'YearBuilt':null, 'YearRemodAdd':null,
       'RoofStyle':null, 'RoofMatl':null, 'Exterior1st':null, 'Exterior2nd':null, 'MasVnrType':null,
       'MasVnrArea':0, 'ExterQual':null, 'ExterCond':null, 'Foundation':null, 'BsmtQual':null,
       'BsmtCond':null, 'BsmtExposure':null, 'BsmtFinType1':null, 'BsmtFinSF1':0,
       'BsmtFinType2':null, 'BsmtFinSF2':0, 'BsmtUnfSF':0, 'TotalBsmtSF':0, 'Heating':null,
       'HeatingQC':null, 'CentralAir':null, 'Electrical':null, '1stFlrSF':0, '2ndFlrSF':0,
       'LowQualFinSF':0, 'GrLivArea':null, 'BsmtFullBath':0, 'BsmtHalfBath':0, 'FullBath':1,
       'HalfBath':0, 'BedroomAbvGr':null, 'KitchenAbvGr':1, 'KitchenQual':null,
       'TotRmsAbvGrd':5, 'Functional':null, 'Fireplaces':0, 'FireplaceQu':null, 'GarageType':null,
       'GarageYrBlt':null, 'GarageFinish':null, 'GarageCars':null, 'GarageArea':0, 'GarageQual':null,
       'GarageCond':null, 'PavedDrive':null, 'WoodDeckSF':0, 'OpenPorchSF':0,
       'EnclosedPorch':0, '3SsnPorch':0, 'ScreenPorch':0, 'PoolArea':0, 'PoolQC':null,
       'Fence':null, 'MiscFeature':null, 'MiscVal':0, 'MoSold':null, 'YrSold':null, 'SaleType':null,
       'SaleCondition':null
        }

        # Parse the form inputs and return the features updated with values entered.
        for key in [k for k in data.keys() if k in feature_values.keys()]:
            feature_values[key] = data[key]

        return feature_values
            
    
    def run(self):
        image = Image.open('images/for_sale.jpg')
        st.image(image, use_column_width=False)
    
    
        add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch'))  
        st.sidebar.info('This app is created to predict sale price of houses' )
        st.sidebar.success('DAT158 - Group 22 - Oblig 2')
        st.title('Prediction of house sale prices')
        
        # Since there are a lot of features in the dataset, I've limited the amount of user input while predicting online.
        # The chosen features are based on the feature_importances of 
        
       
        if add_selectbox == 'Online': 
            overallQual = st.number_input('Overall Quality (1 = Low, 10 = Very Excellent)', min_value=1, max_value=10, value=5)
            overallCond = st.number_input('Overall Condition (1 = Low, 10 = Very Excellent)', min_value=1, max_value=10, value=5)
            grLivArea = st.number_input('Above grade (ground) living area square feet', min_value=0)
            
            # Making display values for some of the basement features
            displayBsmtQual = ['Excellent (100+ inches)', 'Good (90-99 inches)', 'Typical (80-89 inches)', 
                   'Fair (70-79 inches)', 'Poor (<70 inches)']
            displayBsmtCond = ['Excellent', 'Good', 'Typical - slight dampness allowed', 
                   'Fair - dampness or some cracking or settling', 'Poor - Severe cracking, settling, or wetness']
            displayExposure = ['Good Exposure', 'Average Exposure (split levels or foyers typically score average or above)',
                               'Minimum Exposure', 'No Exposure']
            displayBsmtFinishLvl = ['Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters',
                                'Average Rec Room', 'Low Quality', 'Unfinished']
            
            # Actual values of display values
            qualCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
            exposure = ['Gd', 'Av', 'Mn', 'No']
            bsmtFinishLvl = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf']
                            
            totalBsmtSF = bsmtQual = bsmtCond = bsmtExposure = bsmtFT1 = bsmtFSF1 = bsmtFT2 = bsmtFSF2 = bsmtUSF = bsmtFullBath = bsmtHalfBath = None
            if st.checkbox('Basement'):
                totalBsmtSF = st.number_input('Total square feet of basement area', min_value=0)
                bsmtQual = st.selectbox('Basement quality', qualCond,  format_func=lambda x: displayBsmtQual[qualCond.index(x)])
                bsmtCond = st.selectbox('Basement condition', qualCond, format_func=lambda x: displayBsmtCond[qualCond.index(x)])
                bsmtExposure = st.selectbox('Walkout or garden level walls', exposure, 
                                            format_func=lambda x:displayExposure[exposure.index(x)])
                bsmtFT1 = st.selectbox('Rating of basement finished area', bsmtFinishLvl, 
                                       format_func=lambda x:displayBsmtFinishLvl[bsmtFinishLvl.index(x)])
                bsmtFT2 = st.selectbox('Rating of basement finished area )if multiple types)', bsmtFinishLvl,
                                      format_func=lambda x:displayBsmtFinishLvl[bsmtFinishLvl.index(x)])
                bsmtFSF1 = st.number_input('Type 1 finished square feet', min_value=0)
                bsmtFSF2 = st.number_input('Type 2 finished square feet', min_value=0)
                bsmtUSF = st.number_input('Unfinished square feet of basement area', min_value=0)
                bsmtFullBath = st.number_input('Basement full bathrooms', min_value=0)
                bsmtHalfBath = st.number_input('Basement half bathrooms', min_value=0)
            
             # Making display values for some of the garage features
            displayQualCond = ['Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor']
            displayGarageFinish = ['Finished', 'Rough Finished', 'Unfinished']
            displayGarageTypes = ['More than one type of garage', 'Attached to home', 'Basement Garage', 
                                  'Built-In (Garage part of house - typically has room above garage)', 'Car Port', 
                                  'Detached from home']
            
            # Actual values of display values
            garageFinishLvl = ['Fin', 'RFn', 'Unf']
            garageTypes = ['2Types', 'Attachd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd']
            
            garageType = garageYrBlt = garageFinish = garageCars = garageArea = garageQual = garageCond = None
            if st.checkbox('Garage'): 
                garageType = st.selectbox('Garage type', garageTypes, format_func=lambda x: displayGarageTypes[garageTypes.index(x)]  )
                garageYrBlt = st.number_input('Year garage was built', min_value=1700, max_value=2021, value=2000)
                garageFinish = st.selectbox('Garage finish', garageFinishLvl, 
                                            format_func=lambda x: displayGarageFinish[garageFinishLvl.index(x)])
                garageCars = st.number_input('Garage car capacity', min_value=0, max_value=50, value=1)
                garageArea = st.number_input('Garage in square feet', min_value=0)
                garageQual = st.selectbox('Garage quality', qualCond, format_func=lambda x: displayQualCond[qualCond.index(x)])
                garageCond = st.selectbox('Garage condition', qualCond,  format_func=lambda x: displayQualCond[qualCond.index(x)])
            
            kitchen = st.number_input('Kitchens above grade', min_value=0, value=1)
            kitchenQual = st.selectbox('Kitchen Quality', qualCond, format_func=lambda x: displayQualCond[qualCond.index(x)])
            fstFlrSF = st.number_input('First floor square feet', min_value=0)
            sndFlrSF = st.number_input('Second floor square feet', min_value=0)
            lotArea = st.number_input('Lot size in square feet', min_value=0)
            lotFrontage = st.number_input('Linear feet of street connected to property', min_value=0)
            yearBuilt = st.number_input('Original construction date', min_value=1300, max_value=2021, value=1990)
            yearAdd = st.number_input('Remodel data (same as construction date if no remodeling or additions',
                                      min_value=1300, max_value=2021, value=1990)
            openPorchSF = st.number_input('Open porch area in square feet', min_value=0)
            fullBath = st.number_input('Full bathrooms above grade', min_value=0)
            halfBath = st.number_input('Half bathrooms above grade', min_value=0)
            totRmsAbvGrd = st.number_input('Total rooms above grade (does not include bathrooms)', min_value=0)
            bedrooms = st.number_input('Bedrooms above grade (does NOT include basement bedrooms)', min_value=0)
            
            displayNeighborhoods = ['Bloomington Heights', 'Bluestem', 'Briardale','Brookside','Clear Creek', 'College Creek',
                                    'Crawford', 'Edwards', 'Gilbert', 'Iowa DOT and Rail Road', 'Meadow Village', 'Mitchell', 
                                    'North Ames', 'Northridge', 'Northpark Villa', 'Northridge Heights', 'Nortwest Ames',
                                    'Old Town', 'South & West of Iowa State University', 'Sawyer', 'Sawyer West', 'Somerset',
                                    'Stone Brook', 'Timberland', ' Veenker']
            neighborhoods = ['Blmngtn', 'Blueste', 'BrDale','BrkSide','ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert',
                             'IDOTRR', 'MeadowV', 'Mitchel', 'Names', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes','OldTown',
                             'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', ' Veenker']
            
            neighborhood = st.selectbox('Neighborhood', neighborhoods, 
                                        format_func=lambda x: displayNeighborhoods[neighborhoods.index(x)])
            
            displayExterior = ['Asbestos Shingles','Asphalt Shingles', 'Brick Common','Brick Face', 'Cinder Block', 'Cement Board',
                               'Hard Board', 'Imitation Stucco', 'Metal Siding', 'Other', 'Plywood','PreCast', 'Stone','Stucco',
                               'Vinyl Siding','Wood Siding', 'Wood Shingles', 'Not available']
            exterior = ['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc', 'MetalSd','Other','Plywood',
                        'PreCast','Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', None]
            
            exterior1 = st.selectbox('Exterior covering on house', exterior, 
                                        format_func=lambda x: displayExterior[exterior.index(x)])
            exterior2 = st.selectbox('Exterior covering on house (if more than one)', exterior, 
                                        format_func=lambda x: displayExterior[exterior.index(x)])
            exteriorQual = st.selectbox('Quality of material on the exterior', qualCond, format_func=lambda x: displayQualCond[qualCond.index(x)])
            exteriorCond = st.selectbox('Condition of material on the exterior', qualCond,  format_func=lambda x: displayQualCond[qualCond.index(x)])
            
            
      
            
            user_input = {'OverallQual': overallQual, 'OverallCond': overallCond, 'GrLivArea': grLivArea, 
                          'TotalBsmtSF': totalBsmtSF, 'BsmtQual': bsmtQual,'BsmtCond': bsmtCond, 'BsmtExposure': bsmtExposure, 
                          'BsmtFinType1': bsmtFT1, 'BsmtFinSF1': bsmtFSF1, 'BsmtFinType2': bsmtFT2, 'BsmtFinSF2': bsmtFSF2, 
                          'BsmtUnfSF': bsmtUSF, 'BsmtFullBath': bsmtFullBath, 'BsmtHalfBath' : bsmtHalfBath,
                         'GarageType': garageType, 'GarageYrBlt': garageYrBlt, 'GarageFinish': garageFinish, 'GarageCars': garageCars,
                         'GarageArea': garageArea, 'GarageQual': garageQual, 'GarageCond': garageCond,
                         'KitchenAbvGr': kitchen, 'KitchenQual': kitchenQual, '1stFlrSF': fstFlrSF, '2ndFlrSF': sndFlrSF,
                         'LotArea': lotArea, 'LotFrontage': lotFrontage, 'YearBuilt': yearBuilt, 'YearRemodAdd': yearAdd,
                          'OpenPorchSF': openPorchSF, 'FullBath': fullBath, 'HalfBath': halfBath, 'TotRmsAbvGrd':totRmsAbvGrd,
                          'BedroomAbvGr': bedrooms, 'Neighborhood': neighborhood, 'Exterior1st': exterior1, 'Exterior2nd':exterior2,
                          'ExterQual': exteriorQual, 'ExterCond': exteriorCond
                         }
            
            input_dict = self.preprocess(user_input)
            input_df = pd.DataFrame(input_dict, index=[0])
        
            output=''
            
            if st.button('Predict'): 
                output = self.predict(input_df)
                self.store_prediction(output)
                
                output = output['Label'][0].round(2)
                
            
            st.success('Predicted saleprice: {} dollars'.format(output))
            
        if add_selectbox == 'Batch': 
            fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
            if fn is not None: 
                input_df = pd.read_csv(fn)
                predictions = self.predict(input_df)
                st.write(predictions)

            
sa = StreamlitApp()
sa.run()

