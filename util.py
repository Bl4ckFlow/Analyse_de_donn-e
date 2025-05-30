"""
Notes:																																													
* 1 Exabyte = 1012 Megabytes																																													
** 1 terabit = 1’000’000 megabits																																													
*** Individuals aged 10 or older																																													
All data are ITU estimates																																													
Values are rounded to 1 decimal																																													
N/A: Not available																																													
Regions are based on the ITU regions, see: http://www.itu.int/en/ITU-D/Statistics/Pages/definitions/regions.aspx 																																													
																																													
Source:																																													
ITU World Telecommunication/ICT Indicators database																																													
Version November 2024, for Facts and Figures 2024																																													

"""

import pandas as pd 
from openpyxl import load_workbook
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt


def load_data(file_path, sheet_name="By BDT region") :

    variables = [
        "Fixed-telephone subscriptions/Millions",
        "Fixed-broadband subscriptions/Millions",
        "Mobile-cellular telephone subscriptions/Millions",
        "Active mobile-broadband subscriptions/Millions",
        "Population covered by a mobile-cellular network/Millions",
        "Population covered by at least a 3G mobile network/Millions",
        "Population covered by at least an LTE/WiMAX mobile network/Millions",
        "Population covered by at least a 5G mobile network/Millions",
        "Fixed broadband traffic(Exabytes)",
        "Mobile broadband traffic(Exabytes)",
        "International bandwidth usage(Tbits/s**)",
        "Individuals using the Internet(Millions)",
        "Individuals owning a mobile phone***(Millions)"
    ]

    regions = ["Africa", "Americas", "Europe", "Asia-Pacific", "Arab States", "CIS"]

    years = list(range(2005, 2025))

    try:
        wb = load_workbook(file_path, data_only=True)
        sheet = wb[sheet_name] 

        all_data = []

        current_var = None

        for row in sheet.iter_rows(min_col=1, max_col=50):
            cell_value = row[0].value
            
            if cell_value is None:
                continue
                
            if any(var in str(cell_value) for var in variables):
                current_var = str(cell_value)
                continue
                
            if cell_value in regions and current_var:
                year_values = []
                for i in range(1, len(years) + 1): 
                    if i < len(row) and row[i].value is not None:
                        year_values.append(row[i].value)
                    else:
                        year_values.append(np.nan)
                
                # Create entries for each year
                for year_idx, year in enumerate(years):
                    if year_idx < len(year_values):
                        all_data.append({
                            'Region': cell_value,
                            'Year': year,
                            'Variable': current_var,
                            'Value': year_values[year_idx]
                        })
        
        # Convert to DataFrame
        df_long = pd.DataFrame(all_data)
        
        # Pivot to get variables as columns
        df_ml = df_long.pivot_table(
            index=['Region', 'Year'],
            columns='Variable', 
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        # Clean column names
        df_ml.columns.name = None
        
        # Rename columns to shorter names
        column_mapping = {
            "Fixed-telephone subscriptions/Millions": "Fixed_telephone",
            "Fixed-broadband subscriptions/Millions": "Fixed_broadband",
            "Mobile-cellular telephone subscriptions/Millions": "Mobile_cellular_sub",
            "Active mobile-broadband subscriptions/Millions": "Active_mobile_broadband_sub",
            "Population covered by a mobile-cellular network/Millions": "Coverage_mobile_cellular",
            "Population covered by at least a 3G mobile network/Millions": "Coverage_3G",
            "Population covered by at least an LTE/WiMAX mobile network/Millions": "Coverage_LTE_WiMAX",
            "Population covered by at least a 5G mobile network/Millions": "Coverage_5G",
            "Fixed broadband traffic(Exabytes)": "Fixed_broadband_traffic_EB",
            "Mobile broadband traffic(Exabytes)": "Mobile_broadband_traffic_EB",
            "International bandwidth usage(Tbits/s**)": "Intl_bandwidth_Tbps",
            "Individuals using the Internet(Millions)": "Internet_users",
            "Individuals owning a mobile phone***(Millions)": "Mobile_phone_owners"
        }
        
        df_ml = df_ml.rename(columns=column_mapping)
        
        # Sort by Region and Year
        df_ml = df_ml.sort_values(['Region', 'Year']).reset_index(drop=True)
        
        return df_ml
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def eda_summary(df):
    print("🔹 Shape du DataFrame:", df.shape)

    print("\n🔹 Types de données:")
    print(df.dtypes)
    
    print("\n🔹 Informations générales:")
    df.info()
    
    print("\n🔹 Nombre de valeurs NaN par colonne:")
    print(df.isna().sum())
    
    print("\n🔹 Nombre de valeurs nulles par colonne (équivalent de NaNs):")
    print(df.isnull().sum())

    print("\n🔹 Statistiques descriptives globales (sur les colonnes numériques):")
    print(df.describe())

    print("\n🔹 Moyenne par variable (Var) et par année:")
    year_cols = [col for col in df.columns if col.isdigit()]

    stats_by_var_year = df.groupby("Var")[year_cols].agg(['mean', 'median', 'var', 'min', 'max'])

    return stats_by_var_year


def correlation_heatmap(df):
    numeric_cols = [col for col in df.columns if col.isdigit()]
    corr_matrix = df[numeric_cols].astype(float).corr()

    plt.figure(figsize=(14, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("🔍 Heatmap des corrélations entre années")
    plt.show()

def correlation_between_vars(df):
    year_cols = [col for col in df.columns if col.isdigit()]
    
    pivot = df.pivot_table(index='Var', values=year_cols, aggfunc='mean')
    
    corr_matrix = pivot.transpose().corr()  
    
    # 4. Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("🔗 Corrélation entre les différentes variables (Var)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()