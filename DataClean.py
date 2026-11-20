import numpy as np
import pandas as pd

# Load the messy CSV dataset
df = pd.read_csv('messy_supply_chain_logistics_data.csv')
print(df.head(10))

# Helper: strip whitespace + collapse underscores (e.g. 'Rail_Cargo' -> 'Rail Cargo')
def tidy(series):
    return (series.astype(str)
                  .str.replace('_', ' ', regex=False)
                  .str.strip()
                  .str.replace(r'\s+', ' ', regex=True))

# 1. Remove duplicate shipments
df = df.drop_duplicates(subset=['Shipment_ID'], keep='first')

# 2. Clean Shipment_ID & Order_ID (Strip whitespace & Uppercase, fill missing ID)
df['Shipment_ID'] = df['Shipment_ID'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
df['Order_ID'] = df['Order_ID'].astype(str).str.strip().str.upper()

# 3. Clean Origin & Destination -> "City, ST" format
def clean_location(x):
    x = str(x).replace('_', ' ').strip()
    if ',' not in x:
        return x.title()
    city, state = x.split(',', 1)
    return f"{city.strip().title()}, {state.strip().upper()}"

df['Origin'] = df['Origin'].apply(clean_location)
df['Destination'] = df['Destination'].apply(clean_location)

# 4. Clean Carrier (Standardize names, fill missing with 'Unknown')
carrier_map = {
    'TCI FREIGHT': 'TCI Freight',
    'DELHIVERY': 'Delhivery',
    'GATI-KWE': 'Gati-KWE',
    'DHL SUPPLY CHAIN': 'DHL Supply Chain',
    'BLUE DART EXPRESS': 'Blue Dart Express',
    'SAFEXPRESS': 'Safexpress',
}
df['Carrier'] = tidy(df['Carrier']).str.upper().map(carrier_map).fillna('Unknown')

# 5. Clean Transport_Mode
mode_map = {
    'AIR FREIGHT': 'Air Freight',
    'RAIL CARGO': 'Rail Cargo',
    'ROAD - FTL': 'Road - FTL',
    'ROAD - LTL': 'Road - LTL',
    'ROAD': 'Road - Unspecified',   # FTL/LTL cannot be known
}
df['Transport_Mode'] = tidy(df['Transport_Mode']).str.upper().map(mode_map).fillna('Unknown')

# 6. Clean Product_Category (Strip, fix underscores, Title case)
df['Product_Category'] = tidy(df['Product_Category']).str.title()

# 7. Clean numeric columns (Strip ₹, commas, blanks; handle negatives/NaNs)
def to_number(series):
    cleaned = (series.astype(str)
                     .str.replace('₹', '', regex=False)
                     .str.replace(',', '', regex=False)
                     .str.strip()
                     .replace(['', 'nan', 'NaN'], np.nan))
    return pd.to_numeric(cleaned, errors='coerce')

for col in ['Quantity_Units', 'Weight_KG', 'Distance_KM', 'Freight_Cost_INR']:
    df[col] = to_number(df[col])
    df.loc[df[col] <= 0, col] = np.nan            # negative / zero values are invalid
    df[col] = df[col].fillna(df[col].median())    # impute with median

df['Quantity_Units'] = df['Quantity_Units'].round().astype(int)
df['Distance_KM'] = df['Distance_KM'].round().astype(int)
df['Weight_KG'] = df['Weight_KG'].round(2)
df['Freight_Cost_INR'] = df['Freight_Cost_INR'].round(2)

# 8. Parse and Standardize dates into datetime (YYYY-MM-DD)
def parse_date(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip().replace('.', '-').replace('/', '-')
    for fmt in ('%Y-%m-%d', '%d-%m-%Y'):
        try:
            return pd.to_datetime(x, format=fmt)
        except ValueError:
            continue
    return pd.NaT   # 'Pending', impossible dates like 31-02-2026, etc.

for col in ['Ship_Date', 'Estimated_Delivery_Date', 'Actual_Delivery_Date']:
    df[col] = df[col].apply(parse_date)

# 9. Clean Shipment_Status
status_map = {
    'DELIVERED': 'Delivered',
    'DELIVRD': 'Delivered',
    'IN TRANSIT': 'In Transit',
    'DELAYED': 'Delayed',
    'OUT FOR DELIVERY': 'Out for Delivery',
    'CUSTOMS/CHECKPOINT HOLD': 'Customs/Checkpoint Hold',
}
df['Shipment_Status'] = tidy(df['Shipment_Status']).str.upper().map(status_map).fillna('Unknown')

# 10. Clean On_Time_Delivery
ontime_map = {
    'YES': 'Yes', 'Y': 'Yes',
    'NO': 'No', 'LATE': 'No', 'N': 'No',
    'IN PROGRESS': 'In Progress',
}
df['On_Time_Delivery'] = tidy(df['On_Time_Delivery']).str.upper().map(ontime_map).fillna('Unknown')

# 11. Logical check: delivery cannot be before shipping
bad = df['Actual_Delivery_Date'] < df['Ship_Date']
df.loc[bad, 'Actual_Delivery_Date'] = pd.NaT

# Export cleaned dataset
df.to_csv('cleaned_supply_chain_logistics_data.csv', index=False, date_format='%Y-%m-%d')

print(df.info())
print(df.isna().sum())
