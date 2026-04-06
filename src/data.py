# data.py
import pandas as pd
from pathlib import Path

# Load the dataset once
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "agriculture_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Functions to return each specialized DataFrame
def get_core_health_prediction_df():
    return df[["NDVI", "SAVI", "Chlorophyll_Content", "Leaf_Area_Index", "Crop_Stress_Indicator", "Crop_Health_Label"]]

def get_pest_weed_monitoring_df():
    return df[["Pest_Hotspots", "Weed_Coverage", "Pest_Damage", "Multispectral_Images", "Thermal_Images"]]

def get_environmental_stress_analysis_df():
    return df[["Soil_Moisture", "Soil_pH", "Temperature", "Humidity", "Rainfall", "Organic_Matter", "Water_Flow"]]

def get_remote_sensing_UAV_analysis_df():
    return df[["High_Resolution_RGB", "Multispectral_Images", "Thermal_Images", "Spatial_Resolution", "Bounding_Boxes", "Ground_Truth_Segmentation"]]

def get_yield_prediction_df():
    return df[["Expected_Yield", "Crop_Type", "Crop_Growth_Stage", "NDVI", "Canopy_Coverage", "Elevation_Data"]]