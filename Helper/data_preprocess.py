import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle



def prepare_data(path_data):
    # Read data from path
    df = pd.read_csv(path_data)

    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Step 1: Extract width and height into two new columns
    df[['width', 'height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})\s*x\s*(\d{3,4})')

    # Step 2: Combine into a single resolution string like 1920x1080
    df['Extracted_Resolution'] = df.apply(
        lambda row: f"{row['width']}x{row['height']}" if pd.notnull(row['width']) and pd.notnull(
            row['height']) else None,
        axis=1
    )

    # Drop temp columns
    df.drop(columns=['width', 'height'], inplace=True)
    df.rename(columns={'Extracted_Resolution': 'screen_resolution'}, inplace=True)

    # Remove GB and Kg...................................................
    df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

    # CPU handling................................................................
    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

    def fetch_processor(text):
        if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
            return text
        else:
            if text.split()[0] == 'Intel':
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'

    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

    df.drop(columns=['Cpu Name','Cpu'], inplace=True)

    # Convert to string and remove .0....................................................................
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)

    # Normalize units
    df["Memory"] = df["Memory"].str.replace('GB', '', regex=False)
    df["Memory"] = df["Memory"].str.replace('TB', '000', regex=False)

    # Split by '+'
    new = df["Memory"].str.split("+", n=1, expand=True)

    # First memory type
    df["first"] = new[0].str.strip()

    # Second memory type (fill missing with empty string)
    df["second"] = new[1].fillna("").str.strip()

    # Define function to extract size
    def extract_info(mem_str):
        digits = ''.join(filter(str.isdigit, mem_str))
        return int(digits) if digits else 0

    # Apply labels
    df["Layer1HDD"] = df["first"].str.contains("HDD", case=False).astype(int)
    df["Layer1SSD"] = df["first"].str.contains("SSD", case=False).astype(int)
    df["Layer1Hybrid"] = df["first"].str.contains("Hybrid", case=False).astype(int)
    df["Layer1Flash_Storage"] = df["first"].str.contains("Flash Storage", case=False).astype(int)

    df["Layer2HDD"] = df["second"].str.contains("HDD", case=False).astype(int)
    df["Layer2SSD"] = df["second"].str.contains("SSD", case=False).astype(int)
    df["Layer2Hybrid"] = df["second"].str.contains("Hybrid", case=False).astype(int)
    df["Layer2Flash_Storage"] = df["second"].str.contains("Flash Storage", case=False).astype(int)

    # Extract numeric values from 'first' and 'second'
    df["first"] = df["first"].apply(extract_info)
    df["second"] = df["second"].apply(extract_info)

    # Calculate sizes
    df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
    df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]
    df["Hybrid"] = df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"]
    df["Flash_Storage"] = df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"]

    # Drop intermediate columns
    df.drop(columns=[
        'first', 'second',
        'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
        'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'
    ], inplace=True)

    df.drop(columns=['Memory', 'Hybrid', 'Flash_Storage'], inplace=True)

    # GPU category................................................
    df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df.drop(columns=['Gpu'], inplace=True)

    # Operating system category........................
    def cat_os(inp):
        if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
            return 'Windows'
        elif inp == 'macOS' or inp == 'Mac OS X':
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['os'] = df['OpSys'].apply(cat_os)
    df.drop(columns=['OpSys'], inplace=True)

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 'Yes' if 'Touchscreen' in x else 'No')

    df['ips'] = df['ScreenResolution'].apply(lambda x: 'Yes' if 'IPS' in x else 'No')
    df.drop(columns=['ScreenResolution'], inplace=True)

    df.rename(columns={
        'Company': 'company',
        'TypeName': 'type',
        'Inches': 'screen_size',
        'Ram': 'ram',
        'Weight': 'weight',
        'Price': 'price',
        'Cpu brand': 'cpu',
        'HDD': 'hdd',
        'SSD': 'ssd',
        'Gpu brand': 'gpu',
        'Touchscreen': 'touchscreen'
    }, inplace=True)

    df = df[
        ['company', 'type', 'screen_size', 'screen_resolution', 'ips', 'cpu', 'ram', 'ssd', 'hdd', 'gpu', 'touchscreen',
         'os', 'weight', 'price']]

    pickle.dump(df, open('df.pkl', 'wb'))

    X = df.drop(columns=['price'])
    y = np.log(df['price'])

    # return {'text':X,
    #         'label':y
    #         }
    return X, y
