# data.py
import pandas as pd
from pathlib import Path

# Load the dataset once
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "agriculture_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Convert categorical 'Crop_Type' to integer codes
df['Crop_Type'] = df['Crop_Type'].astype('category').cat.codes

# Functions to return each specialized DataFrame
def get_core_health_prediction_df():
    return df[["NDVI", "SAVI", "Chlorophyll_Content", "Leaf_Area_Index", "Crop_Stress_Indicator", "Crop_Health_Label"]]

def get_environmental_stress_analysis_df():
    return df[[
        "Pest_Hotspots",      
        "Pest_Damage",        
        "Thermal_Images",     
        "Soil_Moisture",      
        "Rainfall",           
        "Water_Flow",         
        "Drainage_Features",  
        "Organic_Matter",     
        "Crop_Stress_Indicator"  
    ]]

