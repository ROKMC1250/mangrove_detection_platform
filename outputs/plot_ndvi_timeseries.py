#!/usr/bin/env python3
"""
NDVI Time Series Plotter
========================
Reads CSV files from national park folders and creates beautiful time series plots.
Converts dates to day-of-year (1-365) for consistent x-axis across all plots.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import warnings
import logging

# Suppress all font-related warnings completely
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Try to use Korean fonts if available
def setup_korean_font():
    """Setup Korean font for matplotlib."""
    import matplotlib.font_manager as fm
    
    # List of Korean fonts to try (in order of preference)
    korean_fonts = [
        'NanumGothic',
        'NanumBarunGothic', 
        'Malgun Gothic',
        'Apple SD Gothic Neo',
        'Noto Sans CJK KR',
        'Noto Sans KR',
        'UnDotum',
        'Gulim'
    ]
    
    # Get available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Find the first available Korean font
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"📝 Using Korean font: {font}")
            return True
    
    # Fallback: try to find any font that contains 'Gothic' or 'Nanum'
    for font in available_fonts:
        if 'Gothic' in font or 'Nanum' in font or 'Dotum' in font:
            plt.rcParams['font.family'] = font
            print(f"📝 Using Korean font: {font}")
            return True
    
    print("⚠️ No Korean font found, titles may not display correctly")
    return False

# Setup fonts
setup_korean_font()

# Set up matplotlib style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Color scheme (matching the screenshot style)
LINE_COLOR = '#4A90D9'      # Blue line
FILL_COLOR = '#B8D4F0'      # Light blue fill
MARKER_COLOR = '#2E6CB5'    # Darker blue for markers


def load_csv_files(outputs_dir):
    """Load all CSV files from subdirectories."""
    csv_files = []
    
    for folder in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder)
        if os.path.isdir(folder_path):
            # Find CSV files in this folder
            csv_pattern = os.path.join(folder_path, '*.csv')
            for csv_file in glob.glob(csv_pattern):
                csv_files.append({
                    'name': folder,
                    'path': csv_file
                })
    
    return csv_files


def read_and_process_csv(csv_path):
    """Read CSV and convert dates to day-of-year."""
    df = pd.read_csv(csv_path)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract day of year (1-365/366)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Also keep month-day for labeling
    df['MonthDay'] = df['Date'].dt.strftime('%m-%d')
    
    # Sort by day of year
    df = df.sort_values('DayOfYear')
    
    return df


def create_date_labels():
    """Create month labels for x-axis (Jan, Feb, ..., Dec)."""
    # Day of year for start of each month (non-leap year)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return month_starts, month_names


def plot_ndvi_timeseries(df, park_name, output_path, show_plot=False):
    """Create a beautiful NDVI time series plot with straight lines."""
    
    # Square-ish figure
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    
    # Get data
    x = df['DayOfYear'].values
    y = df['Mean Value'].values
    
    # Sort data
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Extend data to cover full year (1-365)
    x_extended = list(x_sorted)
    y_extended = list(y_sorted)
    
    if x_sorted[0] > 1:
        x_extended.insert(0, 1)
        y_extended.insert(0, y_sorted[0])
    
    if x_sorted[-1] < 365:
        x_extended.append(365)
        y_extended.append(y_sorted[-1])
    
    x_extended = np.array(x_extended)
    y_extended = np.array(y_extended)
    
    # Plot filled area under the straight lines
    ax.fill_between(x_extended, y_extended, alpha=0.4, color=FILL_COLOR, linewidth=0)
    
    # Plot straight lines (thick)
    ax.plot(x_extended, y_extended, color=LINE_COLOR, linewidth=4, zorder=3)
    
    # Plot markers (original data points only, larger)
    ax.scatter(x_sorted, y_sorted, color=MARKER_COLOR, s=80, zorder=4, 
               edgecolors='white', linewidths=2)
    
    # Set x-axis limits: ALWAYS 1 to 365
    ax.set_xlim(1, 365)
    
    # Set y-axis limits with some padding
    y_min = max(0, df['Mean Value'].min() - 0.1)
    y_max = min(1, df['Mean Value'].max() + 0.1)
    ax.set_ylim(y_min, y_max)
    
    # X-axis: show all months
    month_starts, month_names = create_date_labels()
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, rotation=45, ha='right')
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Date', fontsize=11, fontweight='medium')
    ax.set_ylabel('NDVI Mean Value', fontsize=11, fontweight='medium')
    ax.set_title(f'Time Series Analysis - NDVI\n{park_name}', fontsize=13, fontweight='bold', pad=10)
    
    # Add legend
    ax.plot([], [], color=LINE_COLOR, linewidth=2, marker='o', 
            markersize=6, markerfacecolor=MARKER_COLOR, markeredgecolor='white',
            label='NDVI Mean Value')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='lightgray')
    
    # Style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ Saved: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_parks_combined(all_data, output_path, show_plot=False):
    """Create a combined plot with all parks using straight lines."""
    
    # Square-ish figure
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    
    # Color palette for multiple parks
    colors = ['#4A90D9', '#E74C3C', '#2ECC71', '#9B59B6', 
              '#F39C12', '#1ABC9C', '#E91E63', '#00BCD4']
    
    for idx, (park_name, df) in enumerate(all_data.items()):
        color = colors[idx % len(colors)]
        
        x = df['DayOfYear'].values
        y = df['Mean Value'].values
        
        # Sort data
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Extend data to cover full year (1-365)
        x_extended = list(x_sorted)
        y_extended = list(y_sorted)
        
        if x_sorted[0] > 1:
            x_extended.insert(0, 1)
            y_extended.insert(0, y_sorted[0])
        
        if x_sorted[-1] < 365:
            x_extended.append(365)
            y_extended.append(y_sorted[-1])
        
        x_extended = np.array(x_extended)
        y_extended = np.array(y_extended)
        
        # Plot straight lines (thick)
        ax.plot(x_extended, y_extended, color=color, linewidth=3.5, 
                label=park_name, alpha=0.9)
        # Plot markers (original data points only)
        ax.scatter(x_sorted, y_sorted, color=color, s=50, alpha=0.8,
                  edgecolors='white', linewidths=1, zorder=5)
    
    # Set x-axis limits: ALWAYS 1 to 365
    ax.set_xlim(1, 365)
    ax.set_ylim(0, 1)
    
    # X-axis: show all months
    month_starts, month_names = create_date_labels()
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, rotation=45, ha='right')
    
    # Grid
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax.set_xlabel('Date', fontsize=11, fontweight='medium')
    ax.set_ylabel('NDVI Mean Value', fontsize=11, fontweight='medium')
    ax.set_title('Time Series Analysis - NDVI (All National Parks)', 
                 fontsize=14, fontweight='bold', pad=10)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='lightgray', 
              fontsize=9, ncol=2)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Saved combined plot: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_summary_table(all_data, output_path):
    """Create a summary table of all parks' NDVI statistics."""
    
    summary_rows = []
    
    for park_name, df in all_data.items():
        summary_rows.append({
            'Park': park_name,
            'Data Points': len(df),
            'Min NDVI': f"{df['Mean Value'].min():.3f}",
            'Max NDVI': f"{df['Mean Value'].max():.3f}",
            'Mean NDVI': f"{df['Mean Value'].mean():.3f}",
            'Std NDVI': f"{df['Mean Value'].std():.3f}",
            'Date Range': f"{df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Saved summary table: {output_path}")
    
    return summary_df


def main():
    """Main function to process all CSV files and create plots."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = script_dir
    
    print("=" * 60)
    print("NDVI Time Series Plotter")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = load_csv_files(outputs_dir)
    
    if not csv_files:
        print("❌ No CSV files found in subdirectories!")
        return
    
    print(f"\n📁 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f['name']}: {os.path.basename(f['path'])}")
    
    # Process each CSV file
    all_data = {}
    
    print("\n📊 Processing and creating plots...")
    
    for csv_info in csv_files:
        park_name = csv_info['name']
        csv_path = csv_info['path']
        
        try:
            # Read and process data
            df = read_and_process_csv(csv_path)
            all_data[park_name] = df
            
            # Create individual plot
            output_png = os.path.join(
                os.path.dirname(csv_path),
                f'ndvi_timeseries_{park_name}.png'
            )
            plot_ndvi_timeseries(df, park_name, output_png)
            
        except Exception as e:
            print(f"❌ Error processing {park_name}: {e}")
    
    # Create combined plot with all parks
    if all_data:
        combined_output = os.path.join(outputs_dir, 'ndvi_timeseries_all_parks.png')
        plot_all_parks_combined(all_data, combined_output)
        
        # Create summary table
        summary_output = os.path.join(outputs_dir, 'ndvi_summary_table.csv')
        summary_df = create_summary_table(all_data, summary_output)
        
        print("\n📋 Summary Table:")
        print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ All plots generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

