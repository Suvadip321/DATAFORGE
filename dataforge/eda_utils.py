import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union

def get_head_tail(df: pd.DataFrame, n_rows: int = 5, start: bool = True) -> pd.DataFrame:
    if start:
        return df.head(n_rows)
    return df.tail(n_rows)

def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    info = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.count(),
        'Missing Count': df.isnull().sum(),
        'Unique Count': df.nunique()
    })
    info['Missing %'] = round((info['Missing Count'] / len(df)) * 100, 2).astype(str) + '%'
    info = info.reset_index().rename(columns={'index': 'Feature'})
    return info

def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().transpose().fillna('-')

def get_value_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    
    counts = df[column].value_counts(dropna=False).head(top_n) 
    percentage = df[column].value_counts(normalize=True, dropna=False).head(top_n) * 100
    
    result = pd.DataFrame({
        'Count': counts,
        'Percentage': percentage.round(2).astype(str) + '%'
    })
    return result.fillna('N/A')

def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    duplicate_rows = df.duplicated().sum()
    
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    overall_missing_percent = round((total_missing / total_cells) * 100, 2)

    outlier_counts = {}
    for col in df.select_dtypes(include=np.number).columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(np.abs(df[col] - mean) > 3 * std)][col].count()
        outlier_counts[col] = int(outliers)

    return {
        'Total Rows': len(df),
        'Total Columns': df.shape[1],
        'Total Missing Values': int(total_missing),
        'Overall Missing %': f"{overall_missing_percent}%",
        'Duplicate Rows Count': int(duplicate_rows),
        'Outlier Counts (Z-score > 3)': outlier_counts
    }

def plot_univariate_distribution(df: pd.DataFrame, col: str, hue: Optional[str] = None) -> Optional[plt.Figure]:
    if col not in df.columns:
        return None

    if hue and hue not in df.columns:
        hue = None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    is_numerical = pd.api.types.is_numeric_dtype(df[col].dtype)
    unique_values = df[col].nunique(dropna=True)
   
    if is_numerical and unique_values <= 10:
        plot_as_categorical = True
    elif is_numerical:
        plot_as_categorical = False
    else:
        plot_as_categorical = True

    if not plot_as_categorical:
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#1f77b4')
        ax.set_title(f'Distribution of {col} (Numerical)', fontsize=14)
        ax.set_xlabel(col)
        
        if hue:
            ax.text(0.98, 0.98, f'Note: hue "{hue}" not applied to histogram', 
                    transform=ax.transAxes, fontsize=8, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        top_counts = df[col].value_counts().head(15).index 
        df_plot = df[df[col].isin(top_counts)]
        plot_series = df_plot[col].dropna().astype(str) if is_numerical else df_plot[col].dropna()
        hue_series = df_plot[hue] if hue else None
        
        sns.countplot(
            y=plot_series, 
            hue=hue_series,
            ax=ax, 
            order=[str(v) for v in top_counts]
        )
        
        title = f'Value Counts of {col}'
        if hue:
            title += f' by {hue}'
        title += f' (Top {len(top_counts)})'
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(col)
        
    plt.tight_layout()
    return fig

def plot_bivariate_analysis(df: pd.DataFrame, x_col: str, y_col: str, plot_type: str, hue_col: Optional[str] = None) -> Optional[plt.Figure]:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    
    df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if plot_type == 'scatter':
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df_clean, ax=ax)
        elif plot_type == 'boxplot':
            sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=df_clean, ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=df_clean, ax=ax)
        else:
            ax.text(0.5, 0.5, f"Plot type '{plot_type}' not supported.", 
                    horizontalalignment='center', verticalalignment='center')
            
        ax.set_title(f'Bivariate Analysis: {plot_type.title()} Plot ({x_col} vs {y_col})', fontsize=14)
        plt.tight_layout()
        return fig
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Plotting Error: Data mismatch for {plot_type}.", 
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> Optional[plt.Figure]:
    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.empty:
        return None

    corr_matrix = numerical_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='viridis', 
        linewidths=.5, 
        linecolor='black',
        cbar_kws={'label': f'Correlation Coefficient ({method.title()})'},
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    return fig

if __name__ == '__main__':
    test_data = {
        'A_Num': [1, 2, np.nan, 4, 100, 2, 5], 
        'B_Cat': ['Red', 'Blue', 'Red', 'Green', 'Red', 'Blue', 'Red'],
        'C_Group': ['A', 'B', 'A', 'C', 'B', 'A', 'C'], 
        'D_Target': [0, 1, 0, 1, 0, 1, 1]
    }
    test_df = pd.DataFrame(test_data)
    test_df.loc[6] = test_df.loc[1]

    print("Test 1:")
    print(get_head_tail(test_df, start=False))
    print("\nTest 2:")
    print(get_column_info(test_df))
    print("\nTest 3:")
    print(get_descriptive_stats(test_df))
    print("\nTest 4:")
    print(get_value_counts(test_df, column='B_Cat'))
    print("\nTest 5:")
    print(get_data_quality_report(test_df))