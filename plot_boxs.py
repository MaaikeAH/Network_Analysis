import matplotlib.pyplot as plt
import pandas as pd
import os

output_folder = "Output"

# Load centrality measures of nodes
centrality_df = pd.read_excel("centrality_matrix_network_2702.xlsx")

# Define colors for the boxplots
boxplot_colors = ['#FDE725', '#35B779', '#2D708E', 'grey']

# Ensure 'Group' is categorical and maintains the desired order
group_order = ['Hub-Bottlenecks', 'Pure Hubs', 'Bottlenecks', 'Control Proteins']
centrality_df['Group'] = pd.Categorical(centrality_df['Group'], categories=group_order, ordered=True)

# List of centrality measures to plot
centrality_measures = ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Eigenvector Centrality', 'Clustering Coefficient']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Plot each centrality measure
for i, measure in enumerate(centrality_measures):
    if i < len(axes):
        ax = centrality_df.boxplot(column=measure, by='Group', patch_artist=True, widths=0.8,ax=axes[i],
                                        boxprops=dict(color='black', linewidth=1),
                                        medianprops=dict(color='black', linewidth=1.5),
                                        whiskerprops=dict(color='black', linewidth=1),
                                        capprops=dict(color='black', linewidth=1),
                                        flierprops=dict(markerfacecolor='grey', marker='o', markersize=3, linestyle='none'))        
        for patch, color in zip(ax.patches, boxplot_colors):
            patch.set_facecolor(color)
        ax.set_title(measure, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(measure)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)

# Remove the empty subplot
fig.delaxes(axes[-1])

plt.tight_layout(pad=1.0)

plt.suptitle('Centrality Measures by Group', y=1.02)  # Adjust the title position
plt.savefig(os.path.join(output_folder, "centrality_measures_boxplots.png"), dpi=300)
# plt.show()