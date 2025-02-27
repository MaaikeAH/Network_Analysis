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

# Create a boxplot for Degree Centrality
plt.figure(figsize=(8, 6))
ax = centrality_df.boxplot(column='Degree Centrality', by='Group', patch_artist=True)

# Set colors for each box based on their order
for patch, color in zip(ax.patches, boxplot_colors):
    patch.set_facecolor(color)

# Customize the plot
plt.title('Degree Centrality by Group')
plt.suptitle('')  # Suppress the default title to avoid overlap
plt.xlabel('Group')
plt.ylabel('Degree Centrality')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_folder, "degree_centrality_boxplot.png"), dpi=300)
plt.show()