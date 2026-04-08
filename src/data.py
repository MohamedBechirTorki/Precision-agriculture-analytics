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
        "Pest_Hotspots",      # 0.85
        "Pest_Damage",        # 0.74
        "Thermal_Images",     # -0.67
        "Soil_Moisture",      # -0.60
        "Rainfall",           # -0.59
        "Water_Flow",         # -0.57
        "Drainage_Features",  # -0.38
        "Organic_Matter",     # -0.26
        "Crop_Stress_Indicator"  # target
    ]]
def get_remote_sensing_UAV_analysis_df():
    return df[["High_Resolution_RGB", "Multispectral_Images", "Thermal_Images", "Spatial_Resolution", "Bounding_Boxes", "Ground_Truth_Segmentation"]]

def get_yield_prediction_df():
    return df[["Expected_Yield", "Crop_Type", "Crop_Growth_Stage", "NDVI", "Canopy_Coverage", "Elevation_Data"]]


print(df.corr()["Crop_Stress_Indicator"].sort_values())