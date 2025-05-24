import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns





def plot_earnings_distribution_by_gender(male_fares, female_fares, show_plot=True):
    """
    Plot overlapping histograms of fare distribution by gender.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(male_fares, alpha=0.6, label=f'Male (N={len(male_fares)})', bins=30, color='blue')
    ax.hist(female_fares, alpha=0.6, label=f'Female (N={len(female_fares)})', bins=30, color='red')
    
    # Add mean lines
    ax.axvline(np.mean(male_fares), color='blue', linestyle='--', alpha=0.8, label=f'Male Mean: {np.mean(male_fares):.2f}')
    ax.axvline(np.mean(female_fares), color='red', linestyle='--', alpha=0.8, label=f'Female Mean: {np.mean(female_fares):.2f}')
    
    ax.set_xlabel('Total Amount')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Trip Fares by Driver Gender')
    ax.legend()
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_driver_earnings_by_night(per_driver_df, show_plot=True):
    """
    Plot average earnings per driver for night shifts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create color mapping for gender
    colors = ['red' if is_male == 0 else 'blue' for is_male in per_driver_df['isMale']]
    gender_labels = ['Female' if is_male == 0 else 'Male' for is_male in per_driver_df['isMale']]
    
    bars = ax.bar(range(len(per_driver_df)), per_driver_df['mean_total_amount'], color=colors)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Male'),
                      Patch(facecolor='red', label='Female')]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('Driver ID')
    ax.set_ylabel('Average Trip Amount per Night')
    ax.set_title('Average Trip Amount per Driver per Night')
    ax.set_xticks(range(len(per_driver_df)))
    ax.set_xticklabels(per_driver_df['Cab_Driver_ID'], rotation=45)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close()














def plot_earnings_distribution_by_weekend_vs_weekday(weekend_data, weekday_data, show_plot=True):
    """
    Plot comparison between weekend and weekday earnings.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    data_to_plot = [weekend_data, weekday_data]
    labels = ['Weekend', 'Weekday']
    
    ax1.boxplot(data_to_plot, labels=labels)
    ax1.set_ylabel('Total Amount')
    ax1.set_title('Weekend vs Weekday Earnings (Box Plot)')
    
    # Add mean points
    means = [np.mean(weekend_data), np.mean(weekday_data)]
    ax1.scatter([1, 2], means, color='red', s=100, marker='D', label='Mean')
    ax1.legend()
    
    # Histogram overlay
    ax2.hist(weekend_data, alpha=0.6, label=f'Weekend (N={len(weekend_data)})', bins=25, color='orange')
    ax2.hist(weekday_data, alpha=0.6, label=f'Weekday (N={len(weekday_data)})', bins=25, color='green')
    
    ax2.axvline(np.mean(weekend_data), color='orange', linestyle='--', alpha=0.8)
    ax2.axvline(np.mean(weekday_data), color='green', linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Total Amount')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Weekend vs Weekday Earnings Distribution')
    ax2.legend()
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close()









def plot_location_earnings_comparison(location_means_dict, show_plot=True):
    """
    Plot earnings comparison across different groups and locations.
    """
    # Combine all location means into a single DataFrame
    combined_df = pd.DataFrame()
    
    for group_name, means_df in location_means_dict.items():
        means_df_copy = means_df.copy()
        means_df_copy['Group'] = group_name
        combined_df = pd.concat([combined_df, means_df_copy], ignore_index=True)
    
    # Pivot for easier plotting
    pivot_df = combined_df.pivot(index='PickUp_Colombo_ID', columns='Group', values='mean_earnings')
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(pivot_df.index))
    width = 0.15
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(pivot_df.columns)))
    
    for i, (group, color) in enumerate(zip(pivot_df.columns, colors)):
        ax.bar(x + (i * width), pivot_df[group], width, label=group, color=color)
    
    ax.set_xlabel('Pickup Colombo ID')
    ax.set_ylabel('Average Earnings')
    ax.set_title('Average Earnings by Pickup Location and Group')
    ax.set_xticks(x + width * (len(pivot_df.columns) - 1) / 2)
    ax.set_xticklabels(pivot_df.index)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        return fig
    






def plot_tipping_heatmap(tips_by_location, show_plot=True):
    """
    Create a heatmap visualization for tipping patterns by location.
    """
    # Prepare data for heatmap
    tips_matrix = tips_by_location.set_index('PickUp_Colombo_ID')
    
    # Normalize data for better visualization
    normalized_data = tips_matrix.copy()
    for col in tips_matrix.columns:
        normalized_data[col] = (tips_matrix[col] - tips_matrix[col].min()) / (tips_matrix[col].max() - tips_matrix[col].min())
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(normalized_data,
                cmap="RdYlBu_r",
                center=0.5,
                annot=tips_matrix.round(3),
                fmt='.3f',
                cbar_kws={'label': 'Normalized Values'},
                ax=ax)
    
    ax.set_title('Tipping Patterns by Pickup Location', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tipping Metrics', fontsize=12)
    ax.set_ylabel('Pickup Colombo ID', fontsize=12)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close()


