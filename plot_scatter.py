import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from adjustText import adjust_text


output_folder = "Output"

# load centrality measures of nodes
centrality_df = pd.read_excel("centrality_matrix_network_2702.xlsx")

# Define thresholds using mean + 1*SD
degree_threshold = centrality_df['Degree Centrality'].mean() + centrality_df['Degree Centrality'].std()
betweenness_threshold = centrality_df['Betweenness Centrality'].mean() + centrality_df['Betweenness Centrality'].std()

# Classify genes into groups based on thresholds
centrality_df['Group'] = ''
centrality_df.loc[(centrality_df['Degree Centrality'] >= degree_threshold) &
                  (centrality_df['Betweenness Centrality'] >= betweenness_threshold), 'Group'] = 'Hub-Bottleneck'

centrality_df.loc[(centrality_df['Degree Centrality'] >= degree_threshold) &
                  (centrality_df['Betweenness Centrality'] < betweenness_threshold), 'Group'] = 'Hub'

centrality_df.loc[(centrality_df['Degree Centrality'] < degree_threshold) &
                  (centrality_df['Betweenness Centrality'] >= betweenness_threshold), 'Group'] = 'Bottleneck'

# Scatter plot of Degree Centrality vs Betweenness Centrality
plt.figure(figsize=(10, 8))
colors = {'Hub': '#35B779', 'Bottleneck': '#2D708E', 'Hub-Bottleneck': '#FDE725', '': 'grey'}
plt.scatter(centrality_df['Degree Centrality'], centrality_df['Betweenness Centrality'], 
            c=centrality_df['Group'].map(colors), s=40)

# Plot thresholds
plt.axvline(x=degree_threshold, color='black', linestyle='--', linewidth=2)
plt.axhline(y=betweenness_threshold, color='black', linestyle='dotted', linewidth=2)

# Annotate top three genes in each category
# texts = []
# for group, color in colors.items():
#     if group == 'Hub':
#         top_genes = centrality_df[centrality_df['Group'] == group].nlargest(2, 'Degree Centrality')
#     elif group == 'Bottleneck':
#         top_genes = centrality_df[centrality_df['Group'] == group].nlargest(2, 'Degree Centrality')
#     elif group == 'Hub-Bottleneck':
#         top_genes = centrality_df[centrality_df['Group'] == group].nlargest(3, 'Degree Centrality')
#     else:
#         continue

    # for _, row in top_genes.iterrows():
        # plt.annotate(row['Gene'], (row['Degree Centrality'], row['Betweenness Centrality']),
        #              textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='black')
#         texts.append(plt.text(row['Degree Centrality'], row['Betweenness Centrality'], row['Gene'],
#                                 fontsize=12, color='black', ha='center'))

# # Adjust text to avoid overlap
# adjust_text(texts)

plt.xlabel('Degree Centrality')
plt.ylabel('Betweenness Centrality')
plt.title('Gene Classification Based on Centralities')
# plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Hub'),
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=10, label='Bottleneck'),
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Hub-Bottleneck')],
#            loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "scatter_plot.png"), dpi=300)
plt.show()