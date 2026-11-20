import numpy as np
import pandas as pd

# Load the messy CSV dataset
df = pd.read_csv('supply_chain_logistics_data.csv')
print (df.head(10))
# 1. Remove duplicate Shipment IDs
df = df.drop_duplicates(subset=['Shipment_ID'], keep='first')

# 2. Clean Origin, Destination, Carrier, Transport_Mode & Product_Category columns
for column in ['Origin', 'Destination', 'Carrier', 'Transport_Mode', 'Product_Category']:
    df[column] = df[column].astype(str).str.strip()

# 3. Clean & Standardize Shipment_Status and On_Time_Delivery
df['Shipment_Status'] = df['Shipment_Status'].astype(str).str.strip()
df['On_Time_Delivery'] = df['On_Time_Delivery'].astype(str).str.strip().str.title()

# 4. Clean Quantity_Units, Weight_KG, Distance_KM & Freight_Cost_INR
for column in ['Quantity_Units', 'Weight_KG', 'Distance_KM', 'Freight_Cost_INR']:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.loc[df[column] < 0, column] = np.nan

    # Impute missing values with median
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

# 5. Clean and handle unrealistic values in Quantity_Units, Weight_KG & Distance_KM
for column in ['Quantity_Units', 'Weight_KG', 'Distance_KM']:
    df.loc[df[column] == 0, column] = np.nan
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

# 6. Parse and Standardize Ship_Date, Estimated_Delivery_Date & Actual_Delivery_Date into datetime (YYYY-MM-DD)
for column in ['Ship_Date', 'Estimated_Delivery_Date', 'Actual_Delivery_Date']:
    df[column] = pd.to_datetime(df[column], format = '%Y-%m-%d', errors='coerce')

# 7. Clean Shipment_ID and Order_ID columns (Strip whitespace & Uppercase)
df['Shipment_ID'] = df['Shipment_ID'].astype(str).str.strip().str.upper()
df['Order_ID'] = df['Order_ID'].astype(str).str.strip().str.upper()

# Export cleaned dataset
df.to_csv('cleaned_supply_chain_logistics_data.csv', index=False)
