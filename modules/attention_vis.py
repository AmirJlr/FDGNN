import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch


def visualize_dummy_graph_attention(model, feature_names=None, molecule_idx=0, figsize=(12, 10), cmap="plasma", save_path=None):
    """
    Robust, crash-proof attention visualization for your GINGAT model.
    """
    try:
        # --- Step 1: Extract and Validate Data ---
        attn = model.last_attention
        required_keys = ["batch", "edge_index", "alpha"]
        for key in required_keys:
            if key not in attn or attn[key] is None:
                print(f"❌ Missing required key: '{key}' in model.last_attention.")
                return

        batch_vector = attn["batch"].cpu().numpy()
        edge_index = attn["edge_index"].cpu().numpy()
        alpha = attn["alpha"].cpu()

        # Handle multi-head attention
        if alpha.dim() > 1:
            alpha = alpha.mean(dim=0)
        alpha = alpha.numpy()

        # --- CRITICAL: Validate Alignment ---
        if len(alpha) != edge_index.shape[1]:
            print(f"❌ FATAL: Alpha length ({len(alpha)}) != Edge index edges ({edge_index.shape[1]}).")
            print("    This indicates a bug in the model's forward pass. Please check GATConv output handling.")
            return

        # --- Step 2: Determine Structure ---
        if feature_names is None:
            unique_batches = np.unique(batch_vector)
            if len(unique_batches) == 0:
                print("❌ Could not infer graph structure from batch vector.")
                return
            nodes_in_first_graph = np.sum(batch_vector == unique_batches[0])
            num_features = nodes_in_first_graph - 1
            feature_names = [f"Feature_{i+1}" for i in range(num_features)]
        else:
            num_features = len(feature_names)

        total_nodes_per_dummy_graph = num_features + 1
        node_names = ["GNN_Embedding"] + feature_names

        # --- Step 3: Create Mask for Target Molecule ---
        src_batch = batch_vector[edge_index[0]]
        tgt_batch = batch_vector[edge_index[1]]
        mask = (src_batch == molecule_idx) & (tgt_batch == molecule_idx)

        if not np.any(mask):
            print(f"❌ No edges found for molecule_idx={molecule_idx}. Available batch indices: {np.unique(batch_vector)}")
            return

        filtered_edge_index = edge_index[:, mask]
        filtered_alpha = alpha[mask]

        first_node_global_idx = np.where(batch_vector == molecule_idx)[0][0]
        filtered_edge_index = filtered_edge_index - first_node_global_idx

        # --- Step 4: Build Attention Matrix ---
        attn_matrix = np.zeros((total_nodes_per_dummy_graph, total_nodes_per_dummy_graph))
        for i in range(filtered_edge_index.shape[1]):
            src, tgt = filtered_edge_index[:, i]
            attn_matrix[src, tgt] = filtered_alpha[i]

        # --- Step 5: Plot ---
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid", font_scale=1.2)

        ax = sns.heatmap(
            pd.DataFrame(attn_matrix, index=node_names, columns=node_names),
            annot=True,
            fmt=".3f",
            cmap=cmap,
            square=True,
            cbar_kws={"shrink": .8},
            linewidths=.5,
            linecolor='white'
        )

        ax.set_title(f"Dummy Graph Attention Weights (Molecule {molecule_idx})", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Target Node", fontsize=14, fontweight='bold')
        ax.set_ylabel("Source Node", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Attention heatmap saved to: {save_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"❌ Unexpected error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()