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

        abs_data = []
        rel_data = []

        current_var = None

        for row in sheet.iter_rows(min_col=1, max_col=50):
            cell_value = row[0].value

            if cell_value is None:
                continue
                
            if any(var in str(cell_value) for var in variables):
                current_var = str(cell_value)
                continue
                
            if cell_value in regions and current_var:

                """abs"""
                year_values = []
                for i in range(1, len(years) + 1): 
                    if i < len(row) and row[i].value is not None:
                        year_values.append(row[i].value)
                    else:
                        year_values.append(np.nan)
                
                # Create entries for each year
                for year_idx, year in enumerate(years):
                    if year_idx < len(year_values):
                        abs_data.append({
                            'Region': cell_value,
                            'Year': year,
                            'Variable': current_var,
                            'Value': year_values[year_idx]
                        })

                """Relative"""
                year_values = []
                for i in range(len(years) +2, len(years) + 24): 
                    if i < len(row) and row[i].value is not None:
                        year_values.append(row[i].value)
                    else:
                        year_values.append(np.nan)
                
                # Create entries for each year
                for year_idx, year in enumerate(years):
                    if year_idx < len(year_values):
                        rel_data.append({
                            'Region': cell_value,
                            'Year': year,
                            'Variable': current_var,
                            'Value': year_values[year_idx]
                        })
        
        # Convert to DataFrame
        df_long_abs = pd.DataFrame(abs_data)
        df_long_rel = pd.DataFrame(rel_data)

        
        # Pivot to get variables as columns
        df_ml_abs = df_long_abs.pivot_table(
            index=['Region', 'Year'],
            columns='Variable', 
            values='Value',
            aggfunc='first'
        ).reset_index()

        # Pivot as usual
        df_ml_rel = df_long_rel.pivot_table(
            index=['Region', 'Year'],
            columns='Variable', 
            values='Value',
            aggfunc='first'
        ).reset_index()

        
        # Clean column names
        df_ml_abs.columns.name = None
        df_ml_rel.columns.name = None
        
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
        
        df_ml_abs = df_ml_abs.rename(columns=column_mapping)
        df_ml_rel = df_ml_rel.rename(columns=column_mapping)
        
        # Sort by Region and Year
        df_ml_abs = df_ml_abs.sort_values(['Region', 'Year']).reset_index(drop=True)
        df_ml_rel = df_ml_rel.sort_values(['Region', 'Year']).reset_index(drop=True)
        
        return df_ml_abs, df_ml_rel
        
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



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

class TelecomEDA:
    """
    Comprehensive Exploratory Data Analysis and Data Preparation for Telecom Data
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.numeric_cols = None
        self.categorical_cols = None
        self.cleaned_df = None
        
    def basic_info(self):
        """
        Display comprehensive dataset information
        """
        print("="*80)
        print("📊 BASIC DATASET INFORMATION")
        print("="*80)
        
        # Dataset shape
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total cells: {self.df.shape[0] * self.df.shape[1]:,}")
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory Usage: {memory_usage:.2f} MB")
        
        # Data types
        print("\n🔢 DATA TYPES:")
        print(self.df.dtypes.value_counts())
        
        # Column information
        print(f"\n📋 COLUMNS ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col} ({self.df[col].dtype})")
        
        # Basic info
        print("\n📈 DATASET INFO:")
        self.df.info()
        
        return self
    
    def identify_column_types(self):
        """
        Identify numeric and categorical columns
        """
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print("\n🔢 NUMERIC COLUMNS:", self.numeric_cols)
        print("📝 CATEGORICAL COLUMNS:", self.categorical_cols)
        
        return self
    
    def missing_values_analysis(self):
        """
        Comprehensive missing values analysis
        """
        print("\n" + "="*80)
        print("🕳️  MISSING VALUES ANALYSIS")
        print("="*80)
        
        # Missing values count and percentage
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Count', ascending=False)
        
        print("Missing Values Summary:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Missing values heatmap
        if missing_df['Missing_Count'].sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            plt.show()
            
            # Missing patterns
            print(f"\nTotal missing values: {self.df.isnull().sum().sum():,}")
            print(f"Rows with any missing values: {self.df.isnull().any(axis=1).sum():,}")
            print(f"Complete rows: {(~self.df.isnull().any(axis=1)).sum():,}")
        else:
            print("✅ No missing values found!")
        
        return self
    
    def descriptive_statistics(self):
        """
        Comprehensive descriptive statistics
        """
        print("\n" + "="*80)
        print("📊 DESCRIPTIVE STATISTICS")
        print("="*80)
        
        if self.numeric_cols:
            # Basic statistics
            print("BASIC STATISTICS:")
            desc = self.df[self.numeric_cols].describe()
            print(desc)
            
            # Additional statistics
            print("\n📈 ADDITIONAL STATISTICS:")
            additional_stats = pd.DataFrame({
                'Variance': self.df[self.numeric_cols].var(),
                'Std_Dev': self.df[self.numeric_cols].std(),
                'Skewness': self.df[self.numeric_cols].skew(),
                'Kurtosis': self.df[self.numeric_cols].kurtosis(),
                'Range': self.df[self.numeric_cols].max() - self.df[self.numeric_cols].min(),
                'IQR': self.df[self.numeric_cols].quantile(0.75) - self.df[self.numeric_cols].quantile(0.25)
            })
            print(additional_stats)
            
            # Outlier detection using IQR method
            print("\n🚨 OUTLIER DETECTION (IQR Method):")
            for col in self.numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
        
        # Categorical statistics
        if self.categorical_cols:
            print("\n📝 CATEGORICAL VARIABLES SUMMARY:")
            for col in self.categorical_cols:
                print(f"\n{col}:")
                print(f"  Unique values: {self.df[col].nunique()}")
                print(f"  Most frequent: {self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'N/A'}")
                print(f"  Value counts:")
                print(self.df[col].value_counts().head())
        
        return self
    
    def visualizations(self):
        """
        Comprehensive data visualizations
        """
        print("\n" + "="*80)
        print("📊 DATA VISUALIZATIONS")
        print("="*80)
        
        if not self.numeric_cols:
            print("No numeric columns to visualize")
            return self
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribution plots
        n_numeric = len(self.numeric_cols)
        if n_numeric > 0:
            fig, axes = plt.subplots(nrows=(n_numeric+2)//3, ncols=3, figsize=(15, 5*((n_numeric+2)//3)))
            axes = axes.flatten() if n_numeric > 1 else [axes]
            
            for i, col in enumerate(self.numeric_cols):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle('Distribution of Numeric Variables', y=1.02, fontsize=16)
            plt.show()
        
        # 2. Box plots for outlier detection
        if n_numeric > 0:
            fig, axes = plt.subplots(nrows=(n_numeric+2)//3, ncols=3, figsize=(15, 5*((n_numeric+2)//3)))
            axes = axes.flatten() if n_numeric > 1 else [axes]
            
            for i, col in enumerate(self.numeric_cols):
                sns.boxplot(data=self.df, y=col, ax=axes[i])
                axes[i].set_title(f'Box Plot: {col}')
            
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
                
            plt.tight_layout()
            plt.suptitle('Box Plots - Outlier Detection', y=1.02, fontsize=16)
            plt.show()
        
        # 3. Correlation analysis
        if len(self.numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df[self.numeric_cols].corr()
            
            # Heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .5})
            plt.title('Correlation Matrix of Numeric Variables')
            plt.tight_layout()
            plt.show()
            
            # Strong correlations
            print("\n🔗 STRONG CORRELATIONS (>0.7 or <-0.7):")
            strong_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append((correlation_matrix.columns[i], 
                                          correlation_matrix.columns[j], 
                                          corr_val))
            
            if strong_corr:
                for var1, var2, corr in strong_corr:
                    print(f"{var1} ↔ {var2}: {corr:.3f}")
            else:
                print("No strong correlations found")
        
        # 4. Time series plots (if Year column exists)
        if 'Year' in self.df.columns:
            plt.figure(figsize=(15, 10))
            
            # Plot each numeric variable over time by region
            for i, col in enumerate([c for c in self.numeric_cols if c != 'Year']):
                plt.subplot(2, 2, i+1)
                
                if 'Region' in self.df.columns:
                    for region in self.df['Region'].unique():
                        region_data = self.df[self.df['Region'] == region]
                        plt.plot(region_data['Year'], region_data[col], marker='o', label=region)
                else:
                    plt.plot(self.df['Year'], self.df[col], marker='o')
                
                plt.title(f'{col} Over Time')
                plt.xlabel('Year')
                plt.ylabel(col)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if i >= 3:  # Limit to 4 subplots
                    break
            
            plt.tight_layout()
            plt.suptitle('Time Series Analysis', y=1.02, fontsize=16)
            plt.show()
        
        return self
    
    def handle_missing_values(self, strategy='smart'):
        """
        Handle missing values with different strategies
        """
        print("\n" + "="*80)
        print("🔧 HANDLING MISSING VALUES")
        print("="*80)
        
        self.cleaned_df = self.df.copy()
        
        if self.cleaned_df.isnull().sum().sum() == 0:
            print("✅ No missing values to handle!")
            return self
        
        print(f"Missing values before cleaning: {self.cleaned_df.isnull().sum().sum()}")
        
        if strategy == 'smart':
            # Smart strategy based on data patterns
            for col in self.cleaned_df.columns:
                if self.cleaned_df[col].isnull().sum() > 0:
                    if col in self.numeric_cols:
                        missing_pct = self.cleaned_df[col].isnull().sum() / len(self.cleaned_df) * 100
                        
                        if missing_pct > 50:
                            # Too many missing values - drop column
                            print(f"Dropping {col} (>{missing_pct:.1f}% missing)")
                            self.cleaned_df.drop(columns=[col], inplace=True)
                        elif 'Year' in self.cleaned_df.columns and 'Region' in self.cleaned_df.columns:
                            # Use forward fill for time series data
                            print(f"Forward filling {col}")
                            self.cleaned_df[col] = self.cleaned_df.groupby('Region')[col].ffill()
                            # Then backward fill any remaining
                            self.cleaned_df[col] = self.cleaned_df.groupby('Region')[col].bfill()
                            # Finally use median for any remaining
                            self.cleaned_df[col].fillna(self.cleaned_df[col].median(), inplace=True)
                        else:
                            # Use median for numeric variables
                            print(f"Filling {col} with median")
                            self.cleaned_df[col].fillna(self.cleaned_df[col].median(), inplace=True)
                    else:
                        # Use mode for categorical variables
                        print(f"Filling {col} with mode")
                        mode_val = self.cleaned_df[col].mode()
                        if not mode_val.empty:
                            self.cleaned_df[col].fillna(mode_val.iloc[0], inplace=True)
        
        elif strategy == 'knn':
            # KNN Imputation for numeric columns
            if self.numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                self.cleaned_df[self.numeric_cols] = imputer.fit_transform(self.cleaned_df[self.numeric_cols])
                print("Applied KNN imputation to numeric columns")
        
        elif strategy == 'iterative':
            # Iterative imputation
            if self.numeric_cols:
                imputer = IterativeImputer(random_state=42)
                self.cleaned_df[self.numeric_cols] = imputer.fit_transform(self.cleaned_df[self.numeric_cols])
                print("Applied iterative imputation to numeric columns")
        
        print(f"Missing values after cleaning: {self.cleaned_df.isnull().sum().sum()}")
        
        return self
    
    def handle_outliers(self, method='iqr', factor=1.5):
        """
        Handle outliers using different methods
        """
        print("\n" + "="*80)
        print("🚨 HANDLING OUTLIERS")
        print("="*80)
        
        if self.cleaned_df is None:
            self.cleaned_df = self.df.copy()
        
        outliers_removed = 0
        
        for col in self.numeric_cols:
            if col in self.cleaned_df.columns:
                if method == 'iqr':
                    Q1 = self.cleaned_df[col].quantile(0.25)
                    Q3 = self.cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outliers_mask = (self.cleaned_df[col] < lower_bound) | (self.cleaned_df[col] > upper_bound)
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        print(f"{col}: Capping {outliers_count} outliers")
                        # Cap outliers instead of removing
                        self.cleaned_df.loc[self.cleaned_df[col] < lower_bound, col] = lower_bound
                        self.cleaned_df.loc[self.cleaned_df[col] > upper_bound, col] = upper_bound
                        outliers_removed += outliers_count
                
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(self.cleaned_df[col].dropna()))
                    outliers_mask = z_scores > 3
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        print(f"{col}: Found {outliers_count} outliers using Z-score")
                        # Cap using percentiles
                        lower_cap = self.cleaned_df[col].quantile(0.01)
                        upper_cap = self.cleaned_df[col].quantile(0.99)
                        self.cleaned_df[col] = self.cleaned_df[col].clip(lower_cap, upper_cap)
        
        print(f"Total outliers handled: {outliers_removed}")
        return self
    
    def data_transformation(self, scaling_method='standard'):
        """
        Apply data transformations and scaling
        """
        print("\n" + "="*80)
        print("🔄 DATA TRANSFORMATION & SCALING")
        print("="*80)
        
        if self.cleaned_df is None:
            self.cleaned_df = self.df.copy()
        
        # Update numeric columns for cleaned data
        numeric_cols_clean = self.cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns from scaling
        id_cols = ['Year', 'Region'] if 'Region' in self.cleaned_df.columns else ['Year']
        cols_to_scale = [col for col in numeric_cols_clean if col not in id_cols]
        
        if not cols_to_scale:
            print("No columns to scale")
            return self
        
        # Apply scaling
        if scaling_method == 'standard':
            scaler = StandardScaler()
            scaler_name = "Standard Scaler (z-score normalization)"
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaler_name = "Min-Max Scaler (0-1 normalization)"
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            scaler_name = "Robust Scaler (median-based)"
        
        print(f"Applying {scaler_name} to: {cols_to_scale}")
        
        # Fit and transform
        self.cleaned_df[cols_to_scale] = scaler.fit_transform(self.cleaned_df[cols_to_scale])
        
        # Store scaler for later use
        self.scaler = scaler
        self.scaled_columns = cols_to_scale
        
        print("✅ Scaling completed")
        
        return self
    
    def feature_engineering(self):
        """
        Create additional features
        """
        print("\n" + "="*80)
        print("⚙️  FEATURE ENGINEERING")
        print("="*80)
        
        if self.cleaned_df is None:
            self.cleaned_df = self.df.copy()
        
        original_features = len(self.cleaned_df.columns)
        
        # Time-based features
        if 'Year' in self.cleaned_df.columns:
            self.cleaned_df['Years_since_start'] = self.cleaned_df['Year'] - self.cleaned_df['Year'].min()
            print("✅ Added: Years_since_start")
        
        # Growth rates (if we have time series data)
        if 'Year' in self.cleaned_df.columns and 'Region' in self.cleaned_df.columns:
            for col in self.numeric_cols:
                if col in self.cleaned_df.columns and col not in ['Year']:
                    growth_col = f'{col}_growth_rate'
                    self.cleaned_df[growth_col] = self.cleaned_df.groupby('Region')[col].pct_change()
                    print(f"✅ Added: {growth_col}")
        
        # Ratio features (for telecom data)
        telecom_cols = [col for col in self.cleaned_df.columns if any(x in col.lower() for x in ['mobile', 'fixed', 'broadband'])]
        
        if len(telecom_cols) >= 2:
            # Mobile to fixed ratio
            mobile_cols = [col for col in telecom_cols if 'mobile' in col.lower()]
            fixed_cols = [col for col in telecom_cols if 'fixed' in col.lower()]
            
            if mobile_cols and fixed_cols:
                self.cleaned_df['Mobile_to_Fixed_ratio'] = (
                    self.cleaned_df[mobile_cols[0]] / (self.cleaned_df[fixed_cols[0]] + 1)
                )
                print("✅ Added: Mobile_to_Fixed_ratio")
        
        # Total subscriptions
        subscription_cols = [col for col in self.cleaned_df.columns 
                           if any(x in col.lower() for x in ['subscription', 'cellular', 'broadband', 'telephone'])]
        
        if len(subscription_cols) > 1:
            self.cleaned_df['Total_subscriptions'] = self.cleaned_df[subscription_cols].sum(axis=1, skipna=True)
            print("✅ Added: Total_subscriptions")
        
        new_features = len(self.cleaned_df.columns) - original_features
        print(f"\n📈 Created {new_features} new features")
        
        return self
    
    def final_summary(self):
        """
        Provide final summary of the cleaned dataset
        """
        print("\n" + "="*80)
        print("📋 FINAL DATASET SUMMARY")
        print("="*80)
        
        final_df = self.cleaned_df if self.cleaned_df is not None else self.df
        
        print(f"Original shape: {self.original_df.shape}")
        print(f"Final shape: {final_df.shape}")
        print(f"Rows change: {final_df.shape[0] - self.original_df.shape[0]:+d}")
        print(f"Columns change: {final_df.shape[1] - self.original_df.shape[1]:+d}")
        
        print(f"\nMissing values: {final_df.isnull().sum().sum()}")
        print(f"Duplicate rows: {final_df.duplicated().sum()}")
        
        print(f"\nData types:")
        print(final_df.dtypes.value_counts())
        
        print(f"\nMemory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Quality score
        missing_score = max(0, 100 - (final_df.isnull().sum().sum() / final_df.size * 100))
        duplicate_score = max(0, 100 - (final_df.duplicated().sum() / len(final_df) * 100))
        overall_score = (missing_score + duplicate_score) / 2
        
        print(f"\n📊 DATA QUALITY SCORE: {overall_score:.1f}/100")
        print(f"   - Missing values score: {missing_score:.1f}/100")
        print(f"   - Duplicate values score: {duplicate_score:.1f}/100")
        
        return final_df
    
    def run_complete_eda(self, missing_strategy='smart', outlier_method='iqr', scaling_method='standard'):
        """
        Run complete EDA pipeline
        """
        print("🚀 STARTING COMPREHENSIVE EDA & DATA PREPARATION")
        print("="*80)
        
        (self
         .basic_info()
         .identify_column_types()
         .missing_values_analysis()
         .descriptive_statistics()
         .visualizations()
         .handle_missing_values(strategy=missing_strategy)
         .handle_outliers(method=outlier_method)
         .feature_engineering()
         .data_transformation(scaling_method=scaling_method)
        )
        
        cleaned_data = self.final_summary()
        
        print("\n✅ EDA & DATA PREPARATION COMPLETED!")
        print("="*80)
        
        return cleaned_data

# Usage example:
"""
# Load your data
df = your_dataframe_here

# Run comprehensive EDA
eda = TelecomEDA(df)
cleaned_df = eda.run_complete_eda(
    missing_strategy='smart',  # 'smart', 'knn', 'iterative'
    outlier_method='iqr',      # 'iqr', 'zscore'
    scaling_method='standard'   # 'standard', 'minmax', 'robust'
)

# Access the cleaned dataframe
print(cleaned_df.head())

# Access individual components if needed
# eda.basic_info()
# eda.visualizations()
# etc.
"""