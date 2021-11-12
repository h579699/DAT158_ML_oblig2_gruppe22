#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.regression import load_model, predict_model 
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os


# In[2]:


class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('sale_price_pred') 
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
   # Preprocess method insipred by code from predict.py from the Flask-tutorial hospitalapp

   # This method makes sure all features are defined going into the model
    
        null = None
        
        # Setting default values for some of the features, while letting the models imputation method deal with the other potentially 
        # missing values (that are  not retrieved from user)
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
        image = Image.open('../images/for_sale.jpg')
        st.image(image, use_column_width=False)
    
        add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch'))  
        st.sidebar.info('This app is created to predict sale price of houses' )
        st.sidebar.success('DAT158 - Group 22 - Oblig 2')
        st.title('Prediction of house saleprices')
        
        # The model is trained on 67 features. To get the best possible result from the online prediction, the group has chosen
        # to include 66 of these as user inputs (exluding saleprice). The group tried experiementing with fewer features and using
        # default values for the missing features, but the results were not as good as we wanted. 
        
        if add_selectbox == 'Online': 
            overallQual = st.number_input('Overall Quality (1 = Low, 10 = Very Excellent)', min_value=1, max_value=10, value=5)
            overallCond = st.number_input('Overall Condition (1 = Low, 10 = Very Excellent)', min_value=1, max_value=10, value=5)
            grLivArea = st.number_input('Above grade (ground) living area square feet', min_value=0)
            
            # Making display values for some of the basement features
            displayBsmtQual = ['Excellent (100+ inches)', 'Good (90-99 inches)', 'Typical (80-89 inches)', 
                   'Fair (70-79 inches)', 'Poor (<70 inches)', 'No basement']
            displayBsmtCond = ['Excellent', 'Good', 'Typical - slight dampness allowed', 
                   'Fair - dampness or some cracking or settling', 'Poor - Severe cracking, settling, or wetness', 'No basement']
            displayExposure = ['Good Exposure', 'Average Exposure (split levels or foyers typically score average or above)',
                               'Minimum Exposure', 'No Exposure', 'No basement']
            displayBsmtFinishLvl = ['Good Living Quarters', 'Average Living Quarters', 'Below Average Living Quarters',
                                'Average Rec Room', 'Low Quality', 'Unfinished', 'No basement']
            
            # Actual values of display values
            qualCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po', None]
            exposure = ['Gd', 'Av', 'Mn', 'No', None]
            bsmtFinishLvl = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', None]
                            
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
            displayQualCond = ['Excellent', 'Good', 'Typical/Average', 'Fair', 'Poor', 'Not available']
            displayGarageFinish = ['Finished', 'Rough Finished', 'Unfinished', 'No garage']
            displayGarageTypes = ['More than one type of garage', 'Attached to home', 'Basement Garage', 
                                  'Built-In (Garage part of house - typically has room above garage)', 'Car Port', 
                                  'Detached from home', 'No garage']
            
            # Actual values of display values
            garageFinishLvl = ['Fin', 'RFn', 'Unf', None]
            garageTypes = ['2Types', 'Attachd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', None]
            
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
            
            # kitchen = st.number_input('Kitchens above grade', min_value=0, value=1) (not used in model)
            kitchenQual = st.selectbox('Kitchen Quality', qualCond, format_func=lambda x: displayQualCond[qualCond.index(x)])
            fstFlrSF = st.number_input('First floor square feet', min_value=0)
            sndFlrSF = st.number_input('Second floor square feet', min_value=0)
                        
            displayMSSubClass = ['1-story 1946 & newer all styles', '1-story 1945 & older', '1-story w/finished attic all ages',
                                 '1-1/2 story - unfinished all ages', '1-1/2 story finished all ages', '2-story 1946 & newer',
                                 '2-story 1945 & older', '2-1/2 story all ages', 'split or multi-level', 'split foyer',
                                 'duplex - all styles and ages', '1-story planned unit development (PUD) - 1946 & newer', 
                                 '1-1/2 story PUD - all ages', '2-story PUD - 1946 & newer', 'PUD multi level- incl split lev/foyer', 
                                 '2 family conversion - all styles and ages']
            MSSC_values = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]
            
            mssc = st.selectbox('Type of dwelling involved in the sale', MSSC_values, 
                                format_func=lambda x: displayMSSubClass[MSSC_values.index(x)])
           
            displayMZS = ['Agriculture', 'Commercial', 'Floating Village Residental', 'Industrial',
                          'Residential High Density', 'Residential Low Density', 'Residential Low Density Park ', 'Residential Medium Density']
            MSZ_values = ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']
            
            MSZoning = st.selectbox('General zoning classification', MSZ_values, format_func=lambda x: displayMZS[MSZ_values.index(x)])
            
            lotArea = st.number_input('Lot size in square feet', min_value=0)
            lotFrontage = st.number_input('Linear feet of street connected to property', min_value=0)
                        
            displayLot = ['Regular', 'Slightly irregular', 'Moderately Irregular', 'Irregular']
            displayLndc = ['Near Flat/Level', 'Banked', 'Hillside', 'Depression']
            displayLotC = ['Inside lot', 'Corner lot', 'Cul-de-sac', 'Frontage on 2 sides of property', 'Frontage on 3 sides of property']
            displayLaslp = ['Gentle slope', 'Moderate Slope', 'Severe Slope']
            
            lot_values = ['Reg','IR1','IR2','IR3']
            lndc_values = ['lvl', 'Bnk', 'HLS', 'Low']
            lotc_values = ['Inside', 'Corner', ' CulDSac', 'FR2', 'FR3']
            laslp_values = ['Gtl', 'Mod', 'Sev']
            
            lotShape = st.selectbox('General shape of property', lot_values, format_func=lambda x: displayLot[lot_values.index(x)])    
            landContour = st.selectbox('Flatness of the property', lndc_values, format_func=lambda x: displayLndc[lndc_values.index(x)])
            lotConfig = st.selectbox('Lot configuration', lotc_values, format_func=lambda x: displayLotC[lotc_values.index(x)])
            landSlope = st.selectbox('Slope of property', laslp_values, format_func=lambda x: displayLaslp[laslp_values.index(x)])
            
            displayCond1 = ['Adjacent to arterial street', 'Adjacent to feeder street', 'Normal', 
                            'Within 200 of North-South Railroad', 'Adjacent to North-South Railroad',
                            'Near positive off-site feature--park, greenbelt, etc.', 'Adjacent to postive off-site feature',
                            'Within 200 of East-West Railroad', 'Adjacent to East-West Railroad']
            cond1_values = ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
            
            cond1 = st.selectbox('Proximity to various conditions', cond1_values, format_func=lambda x: displayCond1[cond1_values.index(x)])
            

            displayBldgT= ['Single-family Detached', 'Two-family Conversion; originally built as one-family dwelling',
                           'Duplex', 'Townhouse End Unit', 'Townhouse Inside Unit']
            displayHost = ['One story', 'One and one-half story: 2nd level finished', 'One and one-half story: 2nd level unfinished',
                           'Two story', 'Two and one-half story: 2nd level finished', 'Two and one-half story: 2nd level unfinished',
                           'Split Foyer', 'Split Level']
                                       
            bldgT_values = ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI']
            host_values = ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']
                                       
            bldgType = st.selectbox('Type of dwelling', bldgT_values, format_func=lambda x: displayBldgT[bldgT_values.index(x)])
            houseStyle = st.selectbox('Style of dwelling', host_values, format_func=lambda x: displayHost[host_values.index(x)])
                                               
            yearBuilt = st.number_input('Original construction date', min_value=1300, max_value=2021, value=1990)
            yearAdd = st.number_input('Remodel data (same as construction date if no remodeling or additions',
                                      min_value=1300, max_value=2021, value=1990)
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
            exteriorQual = st.selectbox('Quality of material on the exterior', qualCond, 
                                        format_func=lambda x: displayQualCond[qualCond.index(x)])
            exteriorCond = st.selectbox('Condition of material on the exterior', qualCond,  
                                        format_func=lambda x: displayQualCond[qualCond.index(x)])
            
            displayFoundation = ['Brick & Tile', 'Cinder Block', 'Poured Concrete', 'Slab', 'Stone', 'Wood']
            foundationValues = ['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood']
            
            foundation = st.selectbox('Type of foundation', foundationValues,  
                                        format_func=lambda x: displayFoundation[foundationValues.index(x)])
            
            displaySaleType = ['Warranty Deed - Conventional', 'Warranty Deed - Cash', 'Warranty Deed - VA Loan','Home just constructed and sold', 
                               'Court Officer Deed/Estate', 'Contract 15% Down payment regular terms', 'Contract Low Down payment and low interest', 
                               'Contract Low Interest', 'Contract Low Down', 'Other']
            displaySaleCond = ['Normal Sale', 'Abnormal Sale', 'Adjoining Land Purchase', 
                               'Allocation - two linked properties with separate deeds, typically condo with a garage unit',
                               'Sale between family members','Home was not completed when last assessed (associated with New Homes)']
            
            saletypes = ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth']
            saleconditions = ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
            
            saleType = st.selectbox('Type of sale', saletypes, 
                                        format_func=lambda x: displaySaleType[saletypes.index(x)])
            saleCond = st.selectbox('Condition of sale', saleconditions, 
                                        format_func=lambda x: displaySaleCond[saleconditions.index(x)])
            
            yearSold = st.number_input('Year Sold (YYYY)', min_value=1300, max_value=2021, value=2000)
            moSold = st.number_input('Month Sold (M)', min_value=1, max_value=12)
  
            displayFence = ['Good Privacy', 'Minimum Privacy', 'Good Wood', 'Minimum Wood/Wire', 'No fence']
            displayPD = ['Paved', 'Partial Pavement', 'Dirt/Gravel']
            
            fenceTypes = ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None]
            PDvalues = ['Y', 'P', 'N']
            
            fence = st.selectbox('Fence quality', fenceTypes, 
                                format_func=lambda x: displayFence[fenceTypes.index(x)])
            pavedDrive = st.selectbox('Paved driveway', PDvalues, 
                                format_func=lambda x: displayPD[PDvalues.index(x)])
      
            
            woodDeckSF = st.number_input('Wood deck area in square feet', min_value=0)
            openPorchSF = st.number_input('Open porch area in square feet', min_value=0)
            enclosedPorch = st.number_input('Enclosed porch area in square feet', min_value=0)
            screenPorch = st.number_input('Screen porch area in square feet', min_value=0)
            
            displayFPQual = ['Excellent - Exceptional Masonry Fireplace','Good - Masonry Fireplace in main level',
                             'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement',
                             'Fair - Prefabricated Fireplace in basement', 'Poor - Ben Franklin Stove', 'No Fireplace']
            
            fireplaces = st.number_input('Number of fireplaces', min_value=0)
            fireplaceQual = None
            if fireplaces > 0:
                fireplaceQual = st.selectbox('Fireplace quality', qualCond, format_func=lambda x: displayFPQual[qualCond.index(x)])
            
            displayFunctional = ['Typical Functionality', 'Minor deductions 1', 'Minor deductions 2', 'Moderate deductions',
                                 'Major deductions 1', 'Major deductions 2', 'Severely damaged', 'Salvage only']
            displayElectrical = ['Standard Circuit Breakers & Romex', 'Fuse Box over 60 AMP and all Romex wiring (Average)',
                                 '60 AMP Fuse Box and mostly Romex wiring (Fair)',
                                 '60 AMP Fuse Box and mostly knob & tube wiring (Poor)', 'Mixed']
            
            functionalValues = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
            electricalValues = ['SBrkr','FuseA','FuseF', 'FuseP', 'Mix']
            
            functional = st.selectbox('Home functionality (Assume typical unless deductions are warranted)', functionalValues, 
                                      format_func=lambda x: displayFunctional[functionalValues.index(x)])
            electrical = st.selectbox('Electrical system', electricalValues, 
                                      format_func=lambda x: displayElectrical[electricalValues.index(x)])
            heatingQC = st.selectbox('Heating quality and condition', qualCond, format_func=lambda x: displayQualCond[qualCond.index(x)])
            
            centralAir = 'N'
            if st.checkbox('Central air conditioning'):
                centralAir = 'Y' 
                
            displayRoofs = ['Flat', 'Gable', 'Gabrel (Barn)', 'Hip', 'Mansard', 'Shed']
            roofs_values = ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed']
            roofStyle = st.selectbox('Type of rood', roofs_values, format_func=lambda x: displayRoofs[roofs_values.index(x)])
            
            displayMasVT = ['Brick Common', 'Brick Face', 'Cinder Block', 'None', 'Stone']
            masVT_values = ['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone']
            masVnrType = st.selectbox('Masonry veneer type', masVT_values, format_func=lambda x: displayMasVT[masVT_values.index(x)])
            masVnrArea = st.number_input('Masonry veneer area in square feet', min_value=0)
            
            user_input = {'OverallQual': overallQual, 'OverallCond': overallCond, 'GrLivArea': grLivArea, 
                          'TotalBsmtSF': totalBsmtSF, 'BsmtQual': bsmtQual,'BsmtCond': bsmtCond, 'BsmtExposure': bsmtExposure, 
                          'BsmtFinType1': bsmtFT1, 'BsmtFinSF1': bsmtFSF1, 'BsmtFinType2': bsmtFT2, 'BsmtFinSF2': bsmtFSF2, 
                          'BsmtUnfSF': bsmtUSF, 'BsmtFullBath': bsmtFullBath, 'BsmtHalfBath' : bsmtHalfBath,
                          'GarageType': garageType, 'GarageYrBlt': garageYrBlt, 'GarageFinish': garageFinish, 'GarageCars': garageCars,
                          'GarageArea': garageArea, 'GarageQual': garageQual, 'GarageCond': garageCond,
                          'KitchenQual': kitchenQual, '1stFlrSF': fstFlrSF, '2ndFlrSF': sndFlrSF,'Fence':fence, 
                          'LotArea': lotArea, 'LotFrontage': lotFrontage, 'YearBuilt': yearBuilt, 'YearRemodAdd': yearAdd,
                          'FullBath': fullBath, 'HalfBath': halfBath, 'TotRmsAbvGrd':totRmsAbvGrd, 'HeatingQC':heatingQC,
                          'BedroomAbvGr': bedrooms, 'Neighborhood': neighborhood, 'Exterior1st': exterior1, 'Exterior2nd':exterior2,
                          'ExterQual': exteriorQual, 'ExterCond': exteriorCond, 'SaleType': saleType, 'SaleCondition': saleCond, 
                          'MoSold':moSold, 'YrSold':yearSold, 'MSSubClass': mssc, 'MSZoning': MSZoning, 'LotShape': lotShape,
                          'LandContour': landContour, 'LotConfig': lotConfig, 'LandSlope':landSlope, 'Condition1': cond1, 
                          'BldgType': bldgType, 'HouseStyle':houseStyle, 'PavedDrive':pavedDrive, 'Foundation': foundation,
                          'WoodDeckSF':woodDeckSF, 'OpenPorchSF':openPorchSF, 'EnclosedPorch': enclosedPorch, 'ScreenPorch': screenPorch,
                          'Fireplaces': fireplaces, 'FireplaceQu': fireplaceQual, 'Functional': functional, 'Electrical': electrical,
                          'CentralAir':centralAir, 'masVnrType':masVnrType, 'MasVnrArea': masVnrArea, 'RoofStyle': roofStyle
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

