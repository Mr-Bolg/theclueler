# clustering.py
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_distances
from config import DEFAULT_THRESHOLD, MAX_PLACEMENT, GOOD_DISTANCE, BAD_DISTANCE, ALPHA_DISTANCE, BETA_SIZE

def average_pairwise_distance(cluster_embeddings):
    """Compute the average pairwise cosine distance for a given cluster."""
    if len(cluster_embeddings) <= 1:
        return 0
    dist_matrix = cosine_distances(cluster_embeddings)
    np.fill_diagonal(dist_matrix, 0)
    return np.mean(dist_matrix)

def find_representative_clue(cluster_data, embeddings):
    """
    Select the most central (representative) clue in a cluster and return both
    the clue and its associated answer.
    """
    if len(cluster_data) == 1:
        row = cluster_data.iloc[0]
        return row["clue"], row["answerline"]
    distances = cosine_distances(embeddings[cluster_data.index])
    avg_distances = distances.mean(axis=1)
    best_idx = np.argmin(avg_distances)
    rep_row = cluster_data.iloc[best_idx]
    return rep_row["clue"], rep_row["answerline"]

def distance_quality(avg_distance, gamma=2.0):
    """
    Compute a distance quality factor (dq) between 0 and 1 using a nonlinear decrease.
    
    For avg_distance <= GOOD_DISTANCE, dq is 1.
    For avg_distance >= BAD_DISTANCE, dq is 0.
    Otherwise:
       dq = 1 - ((avg_distance - GOOD_DISTANCE) / (BAD_DISTANCE - GOOD_DISTANCE))^gamma
    
    gamma > 1 results in a slower decrease near GOOD_DISTANCE and a faster decrease near BAD_DISTANCE.
    """
    from config import GOOD_DISTANCE, BAD_DISTANCE
    if avg_distance <= GOOD_DISTANCE:
        return 1.0
    elif avg_distance >= BAD_DISTANCE:
        return 0.0
    else:
        ratio = (avg_distance - GOOD_DISTANCE) / (BAD_DISTANCE - GOOD_DISTANCE)
        dq = 1 - (ratio ** gamma)
        return dq

def process_clues_dataframe(df, model, threshold=DEFAULT_THRESHOLD):
    """
    Given a DataFrame with columns: clue, placement, answerline,
    compute embeddings, perform hierarchical clustering using the given threshold,
    compute cluster quality using the old formula, and return a list of clusters.
    
    Each cluster is represented as a tuple:
      (average_placement, cluster_size, avg_distance, quality, rep_clue, rep_answer, full_clues)
    
    Quality = ALPHA_DISTANCE * dq + BETA_SIZE * (cluster_size / max_cluster_size)
    Note: Although placement is computed (for display), it is not used in quality.
    """
    try:
        clues = df["clue"].tolist()
        embeddings = model.encode(clues, batch_size=32, convert_to_numpy=True)
    except Exception as e:
        raise Exception(f"Error during embedding computation: {e}")

    try:
        linkage_matrix = sch.linkage(embeddings, method="ward")
        df["cluster"] = sch.fcluster(linkage_matrix, threshold, criterion="distance")
        max_cluster_size = df["cluster"].value_counts().max()
    except Exception as e:
        raise Exception(f"Error during clustering: {e}")

    cluster_stats = []
    for cluster in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster]
        cluster_embeddings = embeddings[cluster_data.index]
        avg_placement = cluster_data["placement"].mean()  # for display only
        cluster_size = len(cluster_data)
        avg_distance = average_pairwise_distance(cluster_embeddings)
        
        # Compute distance quality factor using nonlinear transformation
        dq = distance_quality(avg_distance, gamma=2.0)
        # Normalize cluster size relative to maximum cluster size
        size_factor = cluster_size / max_cluster_size
        
        # Old cluster quality formula: quality = ALPHA_DISTANCE * dq + BETA_SIZE * size_factor
        CQ = ALPHA_DISTANCE * dq + BETA_SIZE * size_factor
        CQ = round(CQ, 4)
        
        rep_clue, rep_answer = find_representative_clue(cluster_data, embeddings)
        full_clues = cluster_data["clue"].tolist()
        cluster_stats.append((avg_placement, cluster_size, avg_distance, CQ, rep_clue, rep_answer, full_clues))
    
    # Sort clusters by quality (highest first) - quality is at index 3
    cluster_stats.sort(key=lambda x: x[3], reverse=True)
    return cluster_stats
