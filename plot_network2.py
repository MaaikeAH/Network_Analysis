import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

output_folder = "Output"

# load centrality measures of nodes
centrality_df = pd.read_excel("centrality_matrix_network_2702.xlsx")

# load PPI data
ppi_df = pd.read_csv("string_interactions_ataxiagenes+antibodytargets_2711.tsv", sep='\t')

# calculate edges using weight normalization
def calculate_edge_weight_absolute(df):
    """
    Calculate edge weights by normalizing parameters globally and averaging them,
    excluding parameters with only zeros or missing values from the global sum.
    """
    # Define the parameters
    parameters = [
        'neighborhood_on_chromosome',
        'gene_fusion',
        'phylogenetic_cooccurrence',
        'homology',
        'coexpression',
        'experimentally_determined_interaction',
        'database_annotated',
        'automated_textmining',
        'combined_score'
    ]
    
    print("\n--- Step 1: Checking global sums for each parameter ---")
    # Step 1: Compute the global sum for each parameter across all rows, excluding columns with only 0s or NaNs
    global_sums = {}
    for param in parameters:
        if df[param].notna().any() and df[param].sum() != 0:
            global_sums[param] = df[param].sum()
            print(f"Global sum for {param}: {global_sums[param]}")
        else:
            global_sums[param] = 1  # Set to 1 to avoid division by zero
            print(f"Global sum for {param}: {global_sums[param]} (set to 1 because column is empty or has only zeros)")
    
    print("\n--- Step 2: Normalizing each parameter ---")
    # Step 2: Normalize each parameter value by its global sum
    for param in parameters:
        if global_sums[param] != 1:  # Only normalize if there is valid data
            df[f'normalized_{param}'] = df[param] / global_sums[param]
            print(f"Normalized values for {param} (first 5 rows):\n{df[f'normalized_{param}'].head()}")
        else:
            df[f'normalized_{param}'] = 0  # Assign 0 if there was no valid data
            print(f"No valid data for {param}, normalized values set to 0")

    print("\n--- Step 3: Computing final custom weight ---")
    # Step 3: Compute the final weight for each edge by averaging normalized parameters
    df['custom_weight'] = df[[f'normalized_{param}' for param in parameters]].mean(axis=1)
    print(f"Custom weights (first 5 rows):\n{df['custom_weight'].head()}")
    
    print("\n--- Function complete ---")
    return df

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

# Filter ppi_df to include only interactions where both nodes are in the classified groups
filtered_ppi_df = ppi_df[ppi_df['#node1'].isin(centrality_df['Gene']) & ppi_df['node2'].isin(centrality_df['Gene'])]

# Calculate edge weights
filtered_ppi_df = calculate_edge_weight_absolute(filtered_ppi_df)

# Filter nodes to include only those classified as hubs, bottlenecks, or hub-bottlenecks
classified_genes = centrality_df[centrality_df['Group'].isin(['Hub', 'Bottleneck', 'Hub-Bottleneck'])]['Gene']
filtered_ppi_df = filtered_ppi_df[filtered_ppi_df['#node1'].isin(classified_genes) & filtered_ppi_df['node2'].isin(classified_genes)]

# Create the graph
G = nx.from_pandas_edgelist(filtered_ppi_df, '#node1', 'node2', ['custom_weight'])

# Plot network using networkx
node_colors = []
for node in G.nodes():
    if node in centrality_df[centrality_df['Group'] == 'Hub-Bottleneck']['Gene'].values:
        node_colors.append('#FDE725')  # Hub-Bottlenecks in yellow
    elif node in centrality_df[centrality_df['Group'] == 'Hub']['Gene'].values:
        node_colors.append('#35B779')  # Hubs in green
    elif node in centrality_df[centrality_df['Group'] == 'Bottleneck']['Gene'].values:
        node_colors.append('#2D708E')  # Bottlenecks in blue

# Extract edge weights for setting edge thickness
edge_weights = [G[u][v]['custom_weight'] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1  # Normalize edge thickness by max weight
edge_widths = [max(0.5, (weight / max_weight) * 5) for weight in edge_weights]  # Scale edge width

# Adjust node spacing by setting the `k` parameter in the spring layout
pos = nx.spring_layout(G, seed=33, iterations=20, k=0.1)  # Increase `k` to space out nodes

# Visualize the graph with adjusted spacing and reduced font size
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='grey')

# plt.title('Hubs (Red), Bottlenecks (Blue), and Hub-Bottlenecks (Yellow) in the PPI Network')
plt.axis('off')  # Hide axis
plt.tight_layout()  # Adjust layout
plt.savefig(os.path.join(output_folder, "network_plot.png"), dpi=300)
plt.show()