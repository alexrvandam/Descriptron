import pandas as pd
from rouge_score import rouge_scorer
import spacy
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import clip
import torch
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
import pacmap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import logging
import glob
from clip.simple_tokenizer import SimpleTokenizer
import cv2
import json
import warnings
import numpy as np
from matplotlib_venn import venn2  # Added for Venn diagrams
from sklearn.neighbors import NearestNeighbors  # Added for connecting species
from scipy.stats import wilcoxon  # Added for statistical tests
from matplotlib.patches import Patch  # **Added to fix the 'Patch' is not defined error**

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    filename='calculate_rogueV35.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -----------------------------
# Suppress Specific FutureWarnings
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting object dtype arrays on .fillna")
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing `palette` without assigning `hue` is deprecated")

# Initialize the CLIP tokenizer globally
tokenizer = SimpleTokenizer()

# -----------------------------
# 1. Initialize ROUGE Scorer and spaCy
# -----------------------------
def initialize_tools():
    """
    Initialize the ROUGE scorer and load the spaCy English model.
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Load spaCy model with error handling
    try:
        nlp = spacy.load('en_core_web_sm')
        logging.info("SpaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        logging.error("SpaCy model 'en_core_web_sm' not found.")
        print("Error: SpaCy model 'en_core_web_sm' not found.")
        print("Please install it using the following command:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    return scorer, nlp

# -----------------------------
# 2. Define preprocess_text Function
# -----------------------------
def preprocess_text(text):
    """
    Preprocess the input text by:
    - Lowercasing
    - Removing non-alphanumeric characters
    - Removing extra whitespace
    
    Parameters:
    - text (str): The input text string.
    
    Returns:
    - str: The preprocessed text.
    """
    # Lowercase the text
    text = text.lower()
    
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# -----------------------------
# 3. Load Taxonomist TSV Data
# -----------------------------
def load_taxonomist_data(taxo_path):
    """
    Load taxonomist descriptions from a TSV file.
    
    Parameters:
    - taxo_path: Path to the taxonomist TSV file.
    
    Returns:
    - taxo_df: DataFrame containing 'base_name', 'description_taxo', and 'image_path'.
    """
    try:
        taxo_df = pd.read_csv(taxo_path, sep='\t')
        logging.info(f"Loaded Taxonomist data from '{taxo_path}'")
    except Exception as e:
        logging.error(f"Error loading Taxonomist data from '{taxo_path}': {e}")
        print(f"Error loading Taxonomist data from '{taxo_path}': {e}")
        sys.exit(1)
    
    # Check for required columns
    required_columns = {'base_name', 'answer', 'image_path'}
    if not required_columns.issubset(taxo_df.columns):
        logging.error(f"Taxonomist data must contain the following columns: {required_columns}")
        print(f"Error: Taxonomist data must contain the following columns: {required_columns}")
        sys.exit(1)
    
    # Select and rename columns
    taxo_df = taxo_df[['base_name', 'answer', 'image_path']].rename(columns={'answer': 'description_taxo'})
    
    # Standardize 'base_name'
    taxo_df['base_name'] = taxo_df['base_name'].str.lower().str.strip()
    
    return taxo_df

# -----------------------------
# 4. Verify Image Paths
# -----------------------------
def verify_image_paths(taxo_df):
    """
    Verify that all image paths in the DataFrame exist.
    
    Parameters:
    - taxo_df: DataFrame containing 'image_path' column.
    
    Returns:
    - taxo_df: DataFrame with existing image paths.
    """
    existing = taxo_df['image_path'].apply(os.path.exists)
    if not existing.all():
        missing = taxo_df[~existing]
        logging.warning("Some image paths do not exist:")
        logging.warning(f"{missing[['base_name', 'image_path']]}")
        print("Warning: Some image paths do not exist. They will be removed.")
        print(missing[['base_name', 'image_path']])
        taxo_df = taxo_df[existing]
    else:
        logging.info("All image paths verified.")
        print("All image paths verified.")
    return taxo_df

# -----------------------------
# 5. Parse GPT-4 Text File
# -----------------------------
def parse_gpt4_text_file(gpt4_text_path):
    """
    Parse the GPT-4 descriptions from a structured text file.
    Each species description starts with "Species Description for *species_name*:" or without asterisks
    and is separated by '---'.
    
    Parameters:
    - gpt4_text_path: Path to the GPT-4 generated text file.
    
    Returns:
    - gpt4_df: DataFrame containing 'base_name' and 'description_gpt4'.
    """
    if not os.path.exists(gpt4_text_path):
        logging.error(f"GPT-4 text file '{gpt4_text_path}' does not exist.")
        print(f"Error: GPT-4 text file '{gpt4_text_path}' does not exist.")
        sys.exit(1)
    
    with open(gpt4_text_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into individual species descriptions based on '---' separator
    species_sections = content.split('---')
    
    gpt4_records = []
    failed_parses = 0
    
    for section in species_sections:
        section = section.strip()
        if not section:
            continue  # Skip empty sections
        
        # Extract species name using regex
        # Pattern: Species Description for *species_name*: or Species Description for species_name:
        species_name_match = re.search(r"Species Description for\s+\*?([^\*:\n]+)\*?:", section, re.IGNORECASE)
        if not species_name_match:
            logging.warning("Could not extract 'base_name' from a section.")
            print("Warning: Could not extract 'base_name' from a section. Skipping.")
            failed_parses += 1
            continue
        base_name = species_name_match.group(1).strip().lower()
        
        # Extract the description text
        # Find the 'Description:' section, which may be preceded by '**'
        description_match = re.search(r"\*?Description:\*?:?\s*\n\s*\n(.*?)(\n\s*\n|$)", section, re.DOTALL | re.IGNORECASE)
        if not description_match:
            logging.warning(f"Could not find 'Description:' section for '{base_name}'.")
            print(f"Warning: Could not find 'Description:' section for 'base_name' = {base_name}. Skipping.")
            failed_parses += 1
            continue
        description = description_match.group(1).strip()
        
        if not description:
            logging.warning(f"No description found for '{base_name}'.")
            print(f"Warning: No description found for 'base_name' = {base_name}. Skipping.")
            failed_parses += 1
            continue
        
        gpt4_records.append({'base_name': base_name, 'description_gpt4': description})
        logging.info(f"Extracted base_name: '{base_name}'")
    
    gpt4_df = pd.DataFrame(gpt4_records)
    logging.info(f"Parsed {len(gpt4_df)} GPT-4 descriptions.")
    print(f"Parsed {len(gpt4_df)} GPT-4 descriptions from '{gpt4_text_path}'")
    if failed_parses > 0:
        logging.info(f"Failed to parse {failed_parses} species descriptions.")
        print(f"Failed to parse {failed_parses} species descriptions.")
    
    return gpt4_df

# -----------------------------
# 6. Compute ROUGE Scores
# -----------------------------
def compute_rouge_scores(taxo_df, gpt4_df, scorer):
    """
    Compute ROUGE scores between taxonomist and GPT-4 descriptions based on 'base_name'.
    
    Parameters:
    - taxo_df: DataFrame containing taxonomist descriptions.
    - gpt4_df: DataFrame containing GPT-4 descriptions.
    - scorer: Initialized ROUGE scorer.
    
    Returns:
    - rouge_df: DataFrame containing ROUGE scores for each 'base_name'.
    """
    results = []
    
    # Merge DataFrames on 'base_name'
    merged_df = pd.merge(taxo_df, gpt4_df, on='base_name', how='inner')
    logging.info(f"Merged {len(merged_df)} records based on 'base_name'")
    print(f"Merged {len(merged_df)} records based on 'base_name'")
    
    # Iterate through merged DataFrame to compute ROUGE scores
    for idx, row in merged_df.iterrows():
        base_name = row['base_name']
        taxo_desc = row['description_taxo']
        gpt4_desc = row['description_gpt4']
        
        if pd.isna(taxo_desc) or pd.isna(gpt4_desc):
            logging.warning(f"Missing description for '{base_name}'.")
            print(f"Warning: Missing description for '{base_name}'. Skipping.")
            continue
        
        # Compute ROUGE scores
        scores = scorer.score(taxo_desc, gpt4_desc)
        
        # Append scores to results
        results.append({
            'base_name': base_name,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_fmeasure': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_fmeasure': scores['rougeL'].fmeasure,
        })
    
    rouge_df = pd.DataFrame(results)
    return rouge_df

# -----------------------------
# 7. Extract Adjectives and Nouns
# -----------------------------
def extract_adj_nouns(text, nlp):
    """
    Extract adjectives and nouns from a given text using spaCy.
    
    Parameters:
    - text: Input text string.
    - nlp: Loaded spaCy model.
    
    Returns:
    - adj: List of adjectives in lowercase.
    - nouns: List of nouns in lowercase.
    """
    text = preprocess_text(text)
    doc = nlp(text)
    adj = [token.text.lower() for token in doc if token.pos_ == 'ADJ']
    nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN']
    return adj, nouns

# -----------------------------
# 8. Build Adjective-Noun DataFrames
# -----------------------------
def build_adj_noun_df(taxo_df, gpt4_df, nlp):
    """
    Build DataFrames containing adjectives and nouns extracted from descriptions.
    
    Parameters:
    - taxo_df: DataFrame containing taxonomist descriptions.
    - gpt4_df: DataFrame containing GPT-4 descriptions.
    - nlp: Loaded spaCy model.
    
    Returns:
    - taxo_adj_nouns_df: DataFrame with 'base_name', 'adjectives', and 'nouns' for taxonomist.
    - gpt4_adj_nouns_df: DataFrame with 'base_name', 'adjectives', and 'nouns' for GPT-4.
    """
    taxo_adj_nouns = []
    for idx, row in taxo_df.iterrows():
        base_name = row['base_name']
        desc = row['description_taxo']
        if pd.isna(desc):
            logging.warning(f"Missing taxonomist description for 'base_name' = {base_name}. Skipping.")
            print(f"Warning: Missing taxonomist description for '{base_name}'. Skipping.")
            continue
        adj, nouns = extract_adj_nouns(desc, nlp)
        taxo_adj_nouns.append({'base_name': base_name, 'adjectives': adj, 'nouns': nouns})
    taxo_adj_nouns_df = pd.DataFrame(taxo_adj_nouns, columns=['base_name', 'adjectives', 'nouns'])
    
    gpt4_adj_nouns = []
    for idx, row in gpt4_df.iterrows():
        base_name = row['base_name']
        desc = row['description_gpt4']
        if pd.isna(desc):
            logging.warning(f"Missing GPT-4 description for 'base_name' = {base_name}. Skipping.")
            print(f"Warning: Missing GPT-4 description for '{base_name}'. Skipping.")
            continue
        adj, nouns = extract_adj_nouns(desc, nlp)
        gpt4_adj_nouns.append({'base_name': base_name, 'adjectives': adj, 'nouns': nouns})
    gpt4_adj_nouns_df = pd.DataFrame(gpt4_adj_nouns, columns=['base_name', 'adjectives', 'nouns'])
    
    logging.info("Extracted adjectives and nouns from both datasets.")
    print("Extracted adjectives and nouns from both datasets.")
    return taxo_adj_nouns_df, gpt4_adj_nouns_df

# -----------------------------
# 9. Build Co-occurrence Dictionary
# -----------------------------
def build_cooccurrence(dataframe):
    """
    Build a co-occurrence dictionary of adjectives and nouns.
    
    Parameters:
    - dataframe: DataFrame containing 'adjectives' and 'nouns'.
    
    Returns:
    - cooc: Nested dictionary where cooc[adj][noun] = count.
    """
    cooc = defaultdict(lambda: defaultdict(int))
    for idx, row in dataframe.iterrows():
        adj = row['adjectives']
        nouns = row['nouns']
        for a in adj:
            for n in nouns:
                cooc[a][n] += 1
    return cooc

# -----------------------------
# 10. Plot Co-occurrence Network with Enhanced Features
# -----------------------------
def plot_cooccurrence(cooc_dict, title, top_n=30, save=False, save_path=None):
    """
    Plot a co-occurrence network using NetworkX and Matplotlib with:
    - Node colors based on word frequency using a yellow to blue colormap.
    - Fixed node sizes to prevent oversized blobs.
    - Edge colors based on co-occurrence counts using a 'hot' colormap.
    - Separate legends (colorbars) for node frequencies and edge co-occurrence counts.
    - Light gray background for better visibility.
    
    Parameters:
    - cooc_dict: Dictionary of adjective-noun co-occurrences.
    - title: Title of the plot.
    - top_n: Number of top co-occurrences to display.
    - save: Boolean indicating whether to save the plot.
    - save_path: File path to save the plot.
    """
    # Convert to edge list with weights and calculate node frequencies
    edges = []
    node_frequency = defaultdict(int)  # To calculate node frequencies
    for adj, nouns in cooc_dict.items():
        for noun, count in nouns.items():
            edges.append((adj, noun, count))
            node_frequency[adj] += count
            node_frequency[noun] += count
    
    # Sort edges by weight and take top_n
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]
    
    if not edges:
        logging.warning(f"No edges to plot for '{title}'.")
        print(f"No edges to plot for '{title}'.")
        if save and save_path:
            # Create an empty plot
            plt.figure(figsize=(12, 8))
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            logging.info(f"Empty plot saved to '{save_path}'")
            print(f"Empty plot saved to '{save_path}'")
        return
    
    # Create graph
    G = nx.Graph()
    for adj, noun, weight in edges:
        G.add_edge(adj, noun, weight=weight)
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Prepare node colors based on frequency
    frequencies = []
    for node in G.nodes():
        freq = node_frequency[node]
        frequencies.append(freq)
    node_cmap = plt.get_cmap('YlGnBu')  # Yellow to Blue
    norm_node = plt.Normalize(min(frequencies), max(frequencies))
    node_colors = [node_cmap(norm_node(freq)) for freq in frequencies]
    
    # Prepare edge colors based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_cmap = plt.get_cmap('hot')  # 'hot' colormap for edges
    norm_edge = plt.Normalize(min(edge_weights), max(edge_weights))
    edge_colors = [edge_cmap(norm_edge(weight)) for weight in edge_weights]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set background color to light gray
    ax.set_facecolor('lightgray')
    fig.patch.set_facecolor('lightgray')  # Set figure background color
    
    # Draw nodes with colors based on frequency
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=300,  # Fixed node size
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges with colors based on co-occurrence counts
    edges_drawn = nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=2,
        alpha=0.6,
        ax=ax
    )
    
    # Draw labels
    labels = nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif',
        ax=ax
    )
    
    # Create ScalarMappable for node colorbar
    sm_node = plt.cm.ScalarMappable(cmap=node_cmap, norm=norm_node)
    sm_node.set_array(frequencies)
    
    # Create ScalarMappable for edge colorbar
    sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm_edge)
    sm_edge.set_array(edge_weights)
    
    # Add colorbar for nodes
    cbar_node = fig.colorbar(sm_node, ax=ax, shrink=0.5, aspect=10)
    cbar_node.set_label('Word Frequency', rotation=270, labelpad=15)
    
    # Add colorbar for edges
    cbar_edge = fig.colorbar(sm_edge, ax=ax, shrink=0.5, aspect=10)
    cbar_edge.set_label('Co-occurrence Count', rotation=270, labelpad=15)
    
    # Set title and remove axes
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save and save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Plot saved to '{save_path}'")
        print(f"Plot saved to '{save_path}'")
    else:
        plt.show()

# -----------------------------
# 11. Count Word Frequencies
# -----------------------------
def count_words(dataframe, column='adjectives'):
    """
    Count word frequencies in a specified column of the DataFrame.
    
    Parameters:
    - dataframe: DataFrame containing lists of words.
    - column: Column name to count words from ('adjectives' or 'nouns').
    
    Returns:
    - Counter object with word frequencies.
    """
    if column not in dataframe.columns:
        logging.warning(f"Column '{column}' does not exist.")
        print(f"Warning: Column '{column}' does not exist in DataFrame.")
        return Counter()
    
    all_words = []
    for items in dataframe[column]:
        all_words.extend(items)
    return Counter(all_words)

# -----------------------------
# 12. Plot Top Words with Gradient Colors
# -----------------------------
def plot_top_words(counter, title, top_n=20, save=False, save_path=None):
    """
    Plot the top N words from a word frequency counter with a color gradient based on counts.
    
    Parameters:
    - counter: Counter object with word frequencies.
    - title: Title of the plot.
    - top_n: Number of top words to display.
    - save: Boolean indicating whether to save the plot.
    - save_path: File path to save the plot.
    """
    if not counter:
        logging.warning(f"No words to plot for '{title}'.")
        print(f"No words to plot for '{title}'.")
        return
    most_common = counter.most_common(top_n)
    words, counts = zip(*most_common)
    plt.figure(figsize=(10,6))
    
    # Normalize counts for colormap
    norm = plt.Normalize(min(counts), max(counts))
    cmap = plt.cm.viridis
    colors = [cmap(norm(count)) for count in counts]  # Convert to list
    
    # Use matplotlib's barh for per-bar colors to avoid Seaborn palette warnings
    plt.barh(list(words), list(counts), color=colors)
    
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    
    if save and save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Word frequency histogram saved to '{save_path}'")
        print(f"Word frequency histogram saved to '{save_path}'")
    else:
        plt.show()

# -----------------------------
# 13. Save Word Frequencies to CSV
# -----------------------------
def save_word_frequencies(taxo_adj_freq, gpt4_adj_freq, taxo_noun_freq, gpt4_noun_freq, output_path='word_frequency_comparison.csv'):
    """
    Save word frequency comparisons to a CSV file.
    
    Parameters:
    - taxo_adj_freq: Counter object for taxonomist adjectives.
    - gpt4_adj_freq: Counter object for GPT-4 adjectives.
    - taxo_noun_freq: Counter object for taxonomist nouns.
    - gpt4_noun_freq: Counter object for GPT-4 nouns.
    - output_path: Path to save the CSV file.
    """
    freq_df = pd.DataFrame({
        'Taxonomist_Adj': pd.Series(taxo_adj_freq),
        'GPT4_Adj': pd.Series(gpt4_adj_freq),
        'Taxonomist_Noun': pd.Series(taxo_noun_freq),
        'GPT4_Noun': pd.Series(gpt4_noun_freq)
    }).fillna(0).infer_objects()  # Added infer_objects() to handle FutureWarning
    freq_df.to_csv(output_path, index=True)
    logging.info(f"Word frequency comparison saved to '{output_path}'")
    print(f"Word frequency comparison saved to '{output_path}'")

# -----------------------------
# 14. Generate CLIP Embeddings with Sliding Window
# -----------------------------
def generate_clip_embeddings(image_paths, descriptions, device='cpu', max_tokens=70, stride=35):
    """
    Generate CLIP embeddings for images and their descriptions using a sliding window approach.
    
    Parameters:
    - image_paths: List of paths to image files.
    - descriptions: List of text descriptions corresponding to the images.
    - device: 'cpu' or 'cuda' for GPU acceleration.
    - max_tokens: Maximum number of tokens allowed for each window (<=77).
    - stride: Number of tokens to move the window each step.
    
    Returns:
    - image_embeddings: Tensor of image embeddings.
    - text_embeddings: Tensor of aggregated text embeddings.
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Preprocess images
    images = []
    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
            images.append(image)
        except Exception as e:
            logging.error(f"Error loading image '{img_path}': {e}")
            print(f"Error loading image '{img_path}': {e}")
            # Append a zero tensor or handle missing images
            images.append(torch.zeros(1, 3, 224, 224))
    
    images = torch.cat(images).to(device)
    
    # Prepare text windows using CLIP's tokenizer
    all_windows = []
    window_mapping = []  # Keeps track of which description each window belongs to
    
    for idx, desc in enumerate(descriptions):
        if pd.isna(desc):
            logging.warning(f"Missing description at index {idx}.")
            desc = ""
        windows = split_into_windows(desc, max_tokens=max_tokens, stride=stride)
        all_windows.extend(windows)
        window_mapping.extend([idx] * len(windows))
    
    # Tokenize descriptions
    try:
        texts = clip.tokenize(all_windows).to(device)
    except RuntimeError as e:
        logging.error(f"Error tokenizing texts: {e}")
        print(f"Error tokenizing texts: {e}")
        sys.exit(1)
    
    with torch.no_grad():
        text_embeddings = model.encode_text(texts)
    
    # Normalize embeddings
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    # Aggregate embeddings per description (e.g., mean)
    aggregated_text_embeddings = []
    text_embeddings = text_embeddings.cpu().numpy()
    
    # Convert to list of lists for easier aggregation
    window_mapping = np.array(window_mapping)
    text_embeddings = np.array(text_embeddings)
    
    for idx in range(len(descriptions)):
        windows_idx = np.where(window_mapping == idx)[0]
        if len(windows_idx) == 0:
            # Handle descriptions with no windows
            aggregated_emb = np.zeros(text_embeddings.shape[1])
        else:
            # Aggregate embeddings (e.g., mean)
            aggregated_emb = text_embeddings[windows_idx].mean(axis=0)
        aggregated_text_embeddings.append(aggregated_emb)
    
    # Convert aggregated embeddings to tensor
    aggregated_text_embeddings = torch.tensor(np.array(aggregated_text_embeddings))
    aggregated_text_embeddings /= aggregated_text_embeddings.norm(dim=-1, keepdim=True)
    
    return images.cpu(), aggregated_text_embeddings.cpu()

# -----------------------------
# 15. Extract Network Metrics
# -----------------------------
def extract_network_metrics(cooc_dict):
    """
    Extract network metrics from a co-occurrence dictionary.
    
    Parameters:
    - cooc_dict: Dictionary where keys are adjectives and values are dictionaries of nouns with counts.
    
    Returns:
    - metrics: Dictionary containing network metrics.
    """
    G = nx.Graph()
    for adj, nouns in cooc_dict.items():
        for noun, count in nouns.items():
            G.add_edge(adj, noun, weight=count)
    
    if G.number_of_nodes() == 0:
        logging.warning("The graph has no nodes. Network metrics set to None.")
        print("Warning: The graph has no nodes. Network metrics set to None.")
        metrics = {
            'density': None,
            'average_clustering': None,
            'average_degree': None,
        }
        return metrics
    
    density = nx.density(G)
    average_clustering = nx.average_clustering(G)
    degrees = [deg for node, deg in G.degree()]
    average_degree = sum(degrees) / len(degrees) if degrees else 0
    metrics = {
        'density': density,
        'average_clustering': average_clustering,
        'average_degree': average_degree,
    }
    return metrics

# -----------------------------
# 16. Compare Network Metrics
# -----------------------------
def compare_network_metrics(metrics_taxo, metrics_gpt4):
    """
    Compare network metrics between Taxonomist and GPT-4 descriptions using Wilcoxon signed-rank test.
    
    Parameters:
    - metrics_taxo: List of metric dictionaries for Taxonomist descriptions.
    - metrics_gpt4: List of metric dictionaries for GPT-4 descriptions.
    
    Returns:
    - comparison_results: Dictionary containing test statistics and p-values.
    """
    comparison_results = {}
    metric_keys = metrics_taxo[0].keys()
    
    for key in metric_keys:
        taxo_values = [m[key] for m in metrics_taxo]
        gpt4_values = [m[key] for m in metrics_gpt4]
        
        # Check if values are not None
        if all(v is not None for v in taxo_values) and all(v is not None for v in gpt4_values):
            # Perform Wilcoxon signed-rank test
            stat, p = wilcoxon(taxo_values, gpt4_values)
            comparison_results[key] = {'Wilcoxon_stat': stat, 'p-value': p}
        else:
            comparison_results[key] = {'Wilcoxon_stat': None, 'p-value': None}
            logging.warning(f"Cannot perform Wilcoxon test for '{key}' due to missing data.")
            print(f"Warning: Cannot perform Wilcoxon test for '{key}' due to missing data.")
    
    return comparison_results

# -----------------------------
# 17. Summarize Network Metrics
# -----------------------------
def summarize_metrics(metrics_list):
    """
    Summarize metrics by computing mean and standard deviation.
    
    Parameters:
    - metrics_list: List of metric dictionaries.
    
    Returns:
    - summary_df: DataFrame containing summary statistics.
    """
    summary = {}
    for key in metrics_list[0].keys():
        # Filter out None values
        valid_values = [m[key] for m in metrics_list if m[key] is not None]
        if valid_values:
            summary[key] = {
                'mean': pd.Series(valid_values).mean(),
                'std': pd.Series(valid_values).std()
            }
        else:
            summary[key] = {
                'mean': None,
                'std': None
            }
    summary_df = pd.DataFrame(summary).T
    return summary_df

# -----------------------------
# 18. Create Thumbnail With Contour
# -----------------------------
def create_thumbnail(image, mask, thumbnail_size=(80, 80), border_color=None, border_thickness=2):
    """
    Create a thumbnail image with a transparent background and a colored contour that hugs the edge.
    
    Parameters:
    - image (numpy.ndarray): Original image in BGR format.
    - mask (numpy.ndarray): Binary mask where foreground is 1.
    - thumbnail_size (tuple): Desired thumbnail size (width, height).
    - border_color (tuple): RGB color for the contour. If None, no contour is added.
    - border_thickness (int): Thickness of the contour in pixels.
    
    Returns:
    - foreground_rgba (numpy.ndarray): Thumbnail image in RGBA format with transparency and contour.
    """
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]
    
    # Convert BGR to RGB for PIL
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(foreground_rgb)
    pil_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
    pil_img.putalpha(pil_mask)
    
    # Resize thumbnail with PIL
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)
    
    # Add contour if specified
    if border_color:
        # Convert PIL image to numpy array
        thumbnail_np = np.array(pil_img)
        
        # Extract the alpha channel and find contours
        alpha = thumbnail_np[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty image for the contour
        contour_img = np.zeros_like(thumbnail_np)
        
        # Convert border_color from RGB to BGR for OpenCV
        border_color_cv = (border_color[2], border_color[1], border_color[0])
        
        # Draw contours on the contour image
        cv2.drawContours(contour_img, contours, -1, border_color_cv + (255,), thickness=border_thickness)
        
        # Overlay the contour on the thumbnail
        # Use alpha channel from contour_img to determine where to overlay
        contour_alpha = contour_img[:, :, 3] > 0
        thumbnail_np[contour_alpha] = contour_img[contour_alpha]
        
        # Convert back to PIL image
        pil_img = Image.fromarray(thumbnail_np)
    
    # Create a new transparent image
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    offset = ((thumbnail_size[0] - pil_img.size[0]) // 2, (thumbnail_size[1] - pil_img.size[1]) // 2)
    new_img.paste(pil_img, offset, pil_img)
    
    # Convert back to numpy
    foreground_rgba = np.array(new_img)
    return foreground_rgba

# -----------------------------
# 19. Plot PaCMAP with Thumbnails Only and Connect Species
# -----------------------------
def plot_pacmap_with_thumbnails_only_and_connections(embeddings_taxo, embeddings_gpt4, labels, image_paths, base_names, mask_dir, output_dir, category_name, thumbnail_size=(80, 80), thumbnail_offset=(5, 0), n_neighbors=3):
    """
    Plot PaCMAP embeddings with image thumbnails positioned next to each point, without plotting the underlying points.
    Additionally, draw thin lines connecting each species thumbnail to its nearest neighbors and bright green lines connecting
    corresponding Taxonomist and GPT-4 species.
    
    Parameters:
    - embeddings_taxo: Numpy array of Taxonomist text embeddings.
    - embeddings_gpt4: Numpy array of GPT-4 text embeddings.
    - labels: List indicating the type ('Taxonomist' or 'GPT-4') for each embedding.
    - image_paths: List of image file paths corresponding to embeddings.
    - base_names: List of base_names corresponding to each embedding.
    - mask_dir: Directory containing mask files.
    - output_dir: Directory to save the plot.
    - category_name: Category name for the plot title.
    - thumbnail_size: Size of the thumbnails.
    - thumbnail_offset: Tuple indicating the (x, y) offset for thumbnails.
    - n_neighbors: Number of nearest neighbors to connect each species to.
    """
    combined_embeddings = np.vstack((embeddings_taxo, embeddings_gpt4))
    
    # Perform PaCMAP
    pac = pacmap.PaCMAP(n_components=2, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embedding_2d = pac.fit_transform(combined_embeddings)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embedding_2d[:,0],
        'y': embedding_2d[:,1],
        'Type': labels,
        'image_path': image_paths,
        'base_name': base_names
    })
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f'PaCMAP of Text Embeddings with Thumbnails Only ({category_name})')
    ax.set_xlabel('PaCMAP Component 1')
    ax.set_ylabel('PaCMAP Component 2')
    
    # Remove axes ticks for cleaner look
    ax.axis('on')
   # ax.set_xticks(np.linspace(min_x, max_x, num=5))
   # ax.set_yticks(np.linspace(min_y, max_y, num=5))
    ax.set_xticks(np.linspace(embedding_2d[:,0].min(), embedding_2d[:,0].max(), 5))  # 5 ticks on x-axis
    ax.set_yticks(np.linspace(embedding_2d[:,1].min(), embedding_2d[:,1].max(), 5))  # 5 ticks on y-axis
 
    # Add thin lines connecting each species to its nearest neighbors
    # Using sklearn's NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(combined_embeddings)
    distances, indices = nbrs.kneighbors(combined_embeddings)
    
    # Draw lines
    for idx, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # Skip the first neighbor (itself)
            x_values = [embedding_2d[idx, 0], embedding_2d[neighbor, 0]]
            y_values = [embedding_2d[idx, 1], embedding_2d[neighbor, 1]]
            ax.plot(x_values, y_values, color='gray', linewidth=0.5, alpha=0.5)
    
    # Add thumbnails next to points with contours
    for idx, row in plot_df.iterrows():
        img_path = row['image_path']
        description_type = row['Type']
        base_name = row['base_name']
        # Set border color based on type
        if description_type == 'Taxonomist':
            border_color = (0, 0, 255)  # Blue in RGB
        elif description_type == 'GPT-4':
            border_color = (255, 0, 0)  # Red in RGB
        else:
            border_color = None
        # Load corresponding mask
        base_filename = os.path.basename(img_path)
        # Assuming mask filenames follow a specific pattern, adjust if necessary
        mask_pattern = f"binary_mask_{base_filename.split('.')[0]}*.png"
        mask_files = glob.glob(os.path.join(mask_dir, mask_pattern))
        if mask_files:
            mask_path = mask_files[0]  # Take the first match
            mask = cv2.imread(mask_path, 0)
            if mask is not None:
                image = cv2.imread(img_path)
                if image is not None:
                    thumbnail = create_thumbnail(image, mask, thumbnail_size=thumbnail_size, border_color=border_color, border_thickness=1)
                    imagebox = OffsetImage(thumbnail, zoom=1.0)
                    # Apply offset
                    ab = AnnotationBbox(
                        imagebox, 
                        (row['x'], row['y']),
                        frameon=False,
                        xybox=thumbnail_offset,
                        boxcoords="offset points",
                        pad=0.0
                    )
                    ax.add_artist(ab)
    
    # Add bright green lines connecting corresponding Taxonomist and GPT-4 species
    unique_base_names = set(base_names)
    for base in unique_base_names:
        # Get indices for Taxonomist and GPT-4
        taxo_idx = plot_df[(plot_df['base_name'] == base) & (plot_df['Type'] == 'Taxonomist')].index
        gpt4_idx = plot_df[(plot_df['base_name'] == base) & (plot_df['Type'] == 'GPT-4')].index
        if len(taxo_idx) == 1 and len(gpt4_idx) == 1:
            taxo_point = (plot_df.loc[taxo_idx[0], 'x'], plot_df.loc[taxo_idx[0], 'y'])
            gpt4_point = (plot_df.loc[gpt4_idx[0], 'x'], plot_df.loc[gpt4_idx[0], 'y'])
            ax.plot([taxo_point[0], gpt4_point[0]], [taxo_point[1], gpt4_point[1]], 
                    color='limegreen', linewidth=1.5, alpha=0.8)
        else:
            logging.warning(f"Could not find exactly one Taxonomist and one GPT-4 entry for base_name '{base}'.")
            print(f"Warning: Could not find exactly one Taxonomist and one GPT-4 entry for base_name '{base}'. Skipping connecting line.")
    
    # Optionally, adjust plot limits to accommodate thumbnails
    # You might need to experiment with these values
    buffer = max(thumbnail_size) / 100  # Adjust buffer as needed
    x_min, x_max = plot_df['x'].min() - buffer, plot_df['x'].max() + buffer
    y_min, y_max = plot_df['y'].min() - buffer, plot_df['y'].max() + buffer
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Create custom legend handles
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Taxonomist'),
        Patch(facecolor='red', edgecolor='red', label='GPT-4'),
        Patch(facecolor='limegreen', edgecolor='limegreen', label='Taxo-GPT4 Connection')
    ]
    
    # Add legend to the plot
    ax.legend(handles=legend_elements, title='Description Type', loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    
    # Corrected f-string without backslash issue
    pattern = r"\W+"
    filename_part = re.sub(pattern, "_", category_name)
    plot_filename = f'pacmap_embeddings_with_thumbnails_only_connections_{filename_part}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"PaCMAP plot with thumbnails only and connections saved to '{plot_path}'")
    print(f"PaCMAP plot with thumbnails only and connections saved to: {plot_path}")

# -----------------------------
# 20. Load COCO Masks
# -----------------------------
def load_coco_masks(json_path, category_name='entire_forewing'):
    """
    Load masks from a COCO-formatted JSON file for a specific category.
    
    Parameters:
    - json_path: Path to the COCO JSON file.
    - category_name: Name of the category to filter masks.
    
    Returns:
    - masks_dict: Dictionary mapping image filenames to mask file paths.
    """
    masks_dict = {}
    
    if not os.path.exists(json_path):
        logging.error(f"COCO JSON file '{json_path}' does not exist.")
        print(f"Error: COCO JSON file '{json_path}' does not exist.")
        return masks_dict  # Proceed without masks
    
    with open(json_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            print(f"Error: JSON decoding error: {e}")
            return masks_dict
    
    # Create a mapping from image_id to file_name
    image_id_to_filename = {image['id']: image['file_name'] for image in data.get('images', [])}
    logging.info(f"Loaded {len(image_id_to_filename)} images from the JSON.")
    print(f"Loaded {len(image_id_to_filename)} images from the JSON.")
    
    # Create a mapping from category_id to category_name
    category_id_to_name = {category['id']: category['name'] for category in data.get('categories', [])}
    logging.info(f"Loaded {len(category_id_to_name)} categories from the JSON.")
    print(f"Loaded {len(category_id_to_name)} categories from the JSON.")
    
    # Iterate through annotations and process those matching the desired category
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        cat_name = category_id_to_name.get(cat_id, None)
        
        if not cat_name:
            logging.warning(f"Annotation ID {ann.get('id')} has an unknown category_id {cat_id}. Skipping.")
            continue
        
        if cat_name != category_name:
            continue  # Skip annotations not matching the desired category
        
        image_id = ann.get('image_id')
        filename = image_id_to_filename.get(image_id, None)
        
        if not filename:
            logging.warning(f"Annotation ID {ann.get('id')} references an unknown image_id {image_id}. Skipping.")
            continue
        
        # Assuming masks are stored with a specific naming pattern, adjust if necessary
        mask_pattern = f"binary_mask_{filename.split('.')[0]}*.png"
        mask_files = glob.glob(os.path.join(os.path.dirname(json_path), mask_pattern))
        if mask_files:
            masks_dict[filename] = mask_files[0]  # Take the first match
        else:
            logging.warning(f"No mask found for image '{filename}'.")
    
    logging.info(f"Loaded masks for {len(masks_dict)} images.")
    print(f"Loaded masks for {len(masks_dict)} images.")
    
    return masks_dict

# -----------------------------
# 21. Save Embeddings Function
# -----------------------------
def save_embeddings(merged_df, output_dir='embeddings'):
    """
    Save image and text embeddings to files.
    
    Parameters:
    - merged_df: DataFrame containing embeddings.
    - output_dir: Directory to save embedding files.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Example: Save as CSV or pickle. Adjust as needed.
    embeddings_path = os.path.join(output_dir, 'embeddings.csv')
    merged_df.to_csv(embeddings_path, index=False)
    logging.info(f"Embeddings saved to '{embeddings_path}'")
    print(f"Embeddings saved to '{embeddings_path}'")

# -----------------------------
# 22. Split into Windows Function
# -----------------------------
def split_into_windows(text, max_tokens=70, stride=35):
    """
    Split text into overlapping windows based on token count.
    
    Parameters:
    - text (str): The input text string.
    - max_tokens (int): Maximum number of tokens per window (<=77).
    - stride (int): Number of tokens to move the window each step.
    
    Returns:
    - windows (list of str): List of text windows.
    """
    tokens = tokenizer.encode(text)
    windows = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        window_tokens = tokens[start:end]
        window_text = tokenizer.decode(window_tokens)
        windows.append(window_text)
        start += stride
    return windows

# -----------------------------
# 23. Plot ROUGE Scores Heatmap
# -----------------------------
def plot_rouge_heatmap(rouge_df, output_dir, save=False, save_path=None):
    """
    Plot ROUGE scores as a heatmap.
    
    Parameters:
    - rouge_df: DataFrame containing ROUGE scores with 'base_name' and ROUGE columns.
    - output_dir: Directory to save the plot.
    - save: Boolean indicating whether to save the plot.
    - save_path: File path to save the plot.
    """
    # Select f-measure scores
    heatmap_data = rouge_df.set_index('base_name')[['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']]
    
    plt.figure(figsize=(12, 20))  # Adjust size as needed
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', vmin=0, vmax=1)
    plt.title('ROUGE Scores Heatmap')
    plt.xlabel('ROUGE Metric')
    plt.ylabel('Species')
    plt.tight_layout()
    
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"ROUGE heatmap saved to '{save_path}'")
        print(f"ROUGE heatmap saved to '{save_path}'")
    else:
        plt.show()

# -----------------------------
# 24. Plot Venn Diagram Function
# -----------------------------
def plot_venn_diagram(set1, set2, labels, title, save=False, save_path=None):
    """
    Plot a Venn diagram for two sets.
    
    Parameters:
    - set1: First set of items.
    - set2: Second set of items.
    - labels: Tuple of labels for the sets.
    - title: Title of the plot.
    - save: Boolean indicating whether to save the plot.
    - save_path: File path to save the plot.
    """
    plt.figure(figsize=(8,8))
    venn = venn2([set1, set2], set_labels=labels)
    plt.title(title)
    plt.tight_layout()
    
    if save and save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Venn diagram saved to '{save_path}'")
        print(f"Venn diagram saved to '{save_path}'")
    else:
        plt.show()

# -----------------------------
# 25. Main Execution Function
# -----------------------------
def main():
    """
    Main function to execute the script.
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate ROUGE scores and visualize embeddings with thumbnails.")
    parser.add_argument('--taxo_path', type=str, required=True, help='Path to taxonomist TSV file.')
    parser.add_argument('--gpt4_text_path', type=str, required=True, help='Path to GPT-4 generated text file.')
    parser.add_argument('--coco_jsonl_path', type=str, required=False, help='Path to COCO JSON file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--category_name', type=str, default='entire_forewing', help='Category name for masks.')
    parser.add_argument('--thumbnail_size', type=int, nargs=2, default=(80, 80), help='Size of thumbnails (width height).')
    parser.add_argument('--thumbnail_offset', type=int, nargs=2, default=(5, 0), help='Offset for thumbnails (x y).')
    parser.add_argument('--n_neighbors', type=int, default=3, help='Number of nearest neighbors to connect each species to.')
    
    # -----------------------------
    # Added: New Argument for Models CSV
    # -----------------------------
    parser.add_argument('--models_csv', type=str, required=False, help='Path to CSV file containing multiple models data for comparison.')
    
    args = parser.parse_args()
    
    # Initialize tools
    scorer, nlp = initialize_tools()
    
    # Load Taxonomist Data
    taxo_df = load_taxonomist_data(args.taxo_path)
    
    # Verify Image Paths
    taxo_df = verify_image_paths(taxo_df)
    
    # Parse GPT-4 Text File
    gpt4_df = parse_gpt4_text_file(args.gpt4_text_path)
    
    # Load COCO Masks (Optional)
    if args.coco_jsonl_path:
        masks_dict = load_coco_masks(args.coco_jsonl_path, category_name=args.category_name)
    else:
        logging.warning("No COCO JSON path provided. Proceeding without masks.")
        print("Warning: No COCO JSON path provided. Proceeding without masks.")
        masks_dict = {}
    
    # Display base_names
    print("\nTaxonomist base_names:")
    print(taxo_df['base_name'].tolist())
    
    print("\nGPT-4 base_names:")
    print(gpt4_df['base_name'].tolist())
    
    # Check overlapping base_names
    overlapping = set(taxo_df['base_name']).intersection(set(gpt4_df['base_name']))
    print(f"\nNumber of overlapping base_names: {len(overlapping)}")
    logging.info(f"Number of overlapping base_names: {len(overlapping)}")
    
    if len(overlapping) == 0:
        logging.error("No overlapping 'base_name' entries found between Taxonomist and GPT-4 datasets.")
        print("Error: No overlapping 'base_name' entries found between Taxonomist and GPT-4 datasets.")
        sys.exit(1)
    
    # Filter DataFrames to only include overlapping base_names
    taxo_df = taxo_df[taxo_df['base_name'].isin(overlapping)].reset_index(drop=True)
    gpt4_df = gpt4_df[gpt4_df['base_name'].isin(overlapping)].reset_index(drop=True)
    
    # Compute ROUGE scores
    rouge_df = compute_rouge_scores(taxo_df, gpt4_df, scorer)
    rouge_output_path = os.path.join(args.output_dir, 'rouge_scores_comparison.csv')
    rouge_df.to_csv(rouge_output_path, index=False)
    print(f"ROUGE scores saved to '{rouge_output_path}'")
    logging.info(f"ROUGE scores saved to '{rouge_output_path}'")
    
    # Plot ROUGE heatmap
    rouge_heatmap_path = os.path.join(args.output_dir, 'rouge_scores_heatmap.png')
    plot_rouge_heatmap(
        rouge_df=rouge_df,
        output_dir=args.output_dir,
        save=True,
        save_path=rouge_heatmap_path
    )
    
    # Extract adjectives and nouns
    taxo_adj_nouns_df, gpt4_adj_nouns_df = build_adj_noun_df(taxo_df, gpt4_df, nlp)
    
    # Build co-occurrence dictionaries
    taxo_cooc = build_cooccurrence(taxo_adj_nouns_df)
    gpt4_cooc = build_cooccurrence(gpt4_adj_nouns_df)
    
    # Plot co-occurrence networks with enhanced features
    plot_cooccurrence(
        cooc_dict=taxo_cooc,
        title="Taxonomist Description Synteny Map",
        top_n=50,
        save=True,
        save_path=os.path.join(args.output_dir, 'taxonomist_synteny_map.png')
    )
    
    plot_cooccurrence(
        cooc_dict=gpt4_cooc,
        title="GPT-4 Description Synteny Map",
        top_n=50,
        save=True,
        save_path=os.path.join(args.output_dir, 'gpt4_synteny_map.png')
    )
    
    # Compare common and unique edges
    taxo_edges = set((adj, noun) for adj, nouns in taxo_cooc.items() for noun in nouns)
    gpt4_edges = set((adj, noun) for adj, nouns in gpt4_cooc.items() for noun in nouns)
    
    common_edges = taxo_edges.intersection(gpt4_edges)
    unique_taxo = taxo_edges - gpt4_edges
    unique_gpt4 = gpt4_edges - taxo_edges
    
    print(f"\nCommon adjective-noun pairs: {len(common_edges)}")
    print(f"Unique to Taxonomist: {len(unique_taxo)}")
    print(f"Unique to GPT-4: {len(unique_gpt4)}")
    
    logging.info(f"Common adjective-noun pairs: {len(common_edges)}")
    logging.info(f"Unique to Taxonomist: {len(unique_taxo)}")
    logging.info(f"Unique to GPT-4: {len(unique_gpt4)}")
    
    # Plot unique co-occurrences for Taxonomist
    unique_taxo_cooc = defaultdict(lambda: defaultdict(int))
    for adj, noun in unique_taxo:
        unique_taxo_cooc[adj][noun] = taxo_cooc[adj][noun]
    plot_cooccurrence(
        cooc_dict=unique_taxo_cooc,
        title="Taxonomist Unique Synteny Map",
        top_n=30,
        save=True,
        save_path=os.path.join(args.output_dir, 'taxonomist_unique_synteny_map.png')
    )
    
    # Plot unique co-occurrences for GPT-4
    unique_gpt4_cooc = defaultdict(lambda: defaultdict(int))
    for adj, noun in unique_gpt4:
        unique_gpt4_cooc[adj][noun] = gpt4_cooc[adj][noun]
    plot_cooccurrence(
        cooc_dict=unique_gpt4_cooc,
        title="GPT-4 Unique Synteny Map",
        top_n=30,
        save=True,
        save_path=os.path.join(args.output_dir, 'gpt4_unique_synteny_map.png')
    )
    
    # Word Frequency Analysis
    taxo_adj_freq = count_words(taxo_adj_nouns_df, 'adjectives')
    gpt4_adj_freq = count_words(gpt4_adj_nouns_df, 'adjectives')
    taxo_noun_freq = count_words(taxo_adj_nouns_df, 'nouns')
    gpt4_noun_freq = count_words(gpt4_adj_nouns_df, 'nouns')
    
    # Save word frequency comparison
    save_word_frequencies(
        taxo_adj_freq=taxo_adj_freq,
        gpt4_adj_freq=gpt4_adj_freq,
        taxo_noun_freq=taxo_noun_freq,
        gpt4_noun_freq=gpt4_noun_freq,
        output_path=os.path.join(args.output_dir, 'word_frequency_comparison.csv')
    )
    
    # Plot Top Words and Save
    plot_top_words(
        counter=taxo_adj_freq,
        title="Top 20 Adjectives - Taxonomist",
        top_n=20,
        save=True,
        save_path=os.path.join(args.output_dir, 'top_20_adjectives_taxonomist.png')
    )
    
    plot_top_words(
        counter=gpt4_adj_freq,
        title="Top 20 Adjectives - GPT-4",
        top_n=20,
        save=True,
        save_path=os.path.join(args.output_dir, 'top_20_adjectives_gpt4.png')
    )
    
    plot_top_words(
        counter=taxo_noun_freq,
        title="Top 20 Nouns - Taxonomist",
        top_n=20,
        save=True,
        save_path=os.path.join(args.output_dir, 'top_20_nouns_taxonomist.png')
    )
    
    plot_top_words(
        counter=gpt4_noun_freq,
        title="Top 20 Nouns - GPT-4",
        top_n=20,
        save=True,
        save_path=os.path.join(args.output_dir, 'top_20_nouns_gpt4.png')
    )
    
    # --- Plot Venn Diagrams for Nouns and Adjectives ---
    
    # Unique nouns
    taxo_nouns = set(taxo_noun_freq.keys())
    gpt4_nouns = set(gpt4_noun_freq.keys())
    
    # Plot Venn diagram for nouns
    venn_nouns_path = os.path.join(args.output_dir, 'noun_overlap_venn.png')
    plot_venn_diagram(
        set1=taxo_nouns,
        set2=gpt4_nouns,
        labels=('Taxonomist Nouns', 'GPT-4 Nouns'),
        title='Noun Overlap Between Taxonomist and GPT-4',
        save=True,
        save_path=venn_nouns_path
    )
    
    # Unique adjectives
    taxo_adjs = set(taxo_adj_freq.keys())
    gpt4_adjs = set(gpt4_adj_freq.keys())
    
    # Plot Venn diagram for adjectives
    venn_adjs_path = os.path.join(args.output_dir, 'adjective_overlap_venn.png')
    plot_venn_diagram(
        set1=taxo_adjs,
        set2=gpt4_adjs,
        labels=('Taxonomist Adjectives', 'GPT-4 Adjectives'),
        title='Adjective Overlap Between Taxonomist and GPT-4',
        save=True,
        save_path=venn_adjs_path
    )
    
    # --- Statistical Comparison of Network Metrics ---
    
    # Extract network metrics
    metrics_taxo = [extract_network_metrics(taxo_cooc)]
    metrics_gpt4 = [extract_network_metrics(gpt4_cooc)]
    
    # Summarize metrics
    summary_taxo = summarize_metrics(metrics_taxo)
    summary_gpt4 = summarize_metrics(metrics_gpt4)
    
    print("\nTaxonomist Metrics Summary:")
    print(summary_taxo)
    
    print("\nGPT-4 Metrics Summary:")
    print(summary_gpt4)
    
    logging.info("Taxonomist Metrics Summary:")
    logging.info(f"\n{summary_taxo}")
    
    logging.info("GPT-4 Metrics Summary:")
    logging.info(f"\n{summary_gpt4}")
    
    # Compare metrics using Wilcoxon signed-rank test
    comparison_results = compare_network_metrics(metrics_taxo, metrics_gpt4)
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_output_path = os.path.join(args.output_dir, 'network_metrics_comparison.csv')
    comparison_df.to_csv(comparison_output_path)
    logging.info("Network metrics comparison saved to 'network_metrics_comparison.csv'")
    print("Network metrics comparison saved to 'network_metrics_comparison.csv'")
    print("\nWilcoxon Signed-Rank Test Results:")
    print(comparison_df)
    
    # --- Embedding-Based Comparison ---
    
    # Merge DataFrames for embeddings
    merged_df = pd.merge(taxo_df, gpt4_df, on='base_name', how='inner')
    if len(merged_df) == 0:
        logging.error("No records to process for embeddings since merged_df is empty.")
        print("Error: No records to process for embeddings since merged_df is empty.")
        sys.exit(1)
    
    # Generate CLIP embeddings with sliding window
    image_paths = merged_df['image_path'].tolist()
    descriptions_taxo = merged_df['description_taxo'].tolist()
    descriptions_gpt4 = merged_df['description_gpt4'].tolist()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")
    
    image_emb_taxo, text_emb_taxo = generate_clip_embeddings(
        image_paths,
        descriptions_taxo,
        device=device,
        max_tokens=70,  # Set to <=77
        stride=35       # Example stride
    )
    image_emb_gpt4, text_emb_gpt4 = generate_clip_embeddings(
        image_paths,
        descriptions_gpt4,
        device=device,
        max_tokens=70,  # Set to <=77
        stride=35       # Example stride
    )
    
    # Add embeddings to DataFrame
    merged_df['image_embedding_taxo'] = list(image_emb_taxo.numpy())
    merged_df['text_embedding_taxo'] = list(text_emb_taxo.numpy())
    merged_df['image_embedding_gpt4'] = list(image_emb_gpt4.numpy())
    merged_df['text_embedding_gpt4'] = list(text_emb_gpt4.numpy())
    
    # Save embeddings
    save_embeddings(merged_df, output_dir=os.path.join(args.output_dir, 'embeddings'))
    
    # Compare embeddings using cosine similarity
    similarities = cosine_similarity(
        np.vstack(merged_df['text_embedding_taxo'].values),
        np.vstack(merged_df['text_embedding_gpt4'].values)
    ).diagonal()
    merged_df['cosine_similarity'] = similarities
    
    # Save cosine similarities
    cosine_sim_output = os.path.join(args.output_dir, 'cosine_similarities.csv')
    merged_df[['base_name', 'cosine_similarity']].to_csv(cosine_sim_output, index=False)
    logging.info(f"Cosine similarities saved to '{cosine_sim_output}'")
    print(f"Cosine similarities saved to '{cosine_sim_output}'")
    
    # Plot distribution of cosine similarities
    plt.figure(figsize=(8,6))
    sns.histplot(merged_df['cosine_similarity'], bins=10, kde=True, color='blue')
    plt.title('Distribution of Cosine Similarities Between Taxonomist and GPT-4 Descriptions')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    cosine_dist_plot = os.path.join(args.output_dir, 'cosine_similarity_distribution.png')
    plt.savefig(cosine_dist_plot, dpi=300)
    plt.close()
    logging.info(f"Cosine similarity distribution plot saved to '{cosine_dist_plot}'")
    print(f"Cosine similarity distribution plot saved to '{cosine_dist_plot}'")
    
    # --- PaCMAP Visualization with Thumbnails Only and Connections ---
    
    # Prepare labels based on description type
    labels = ['Taxonomist'] * len(text_emb_taxo) + ['GPT-4'] * len(text_emb_gpt4)
    
    # Prepare base_names duplicated for Taxonomist and GPT-4
    base_names_taxo = merged_df['base_name'].tolist()
    base_names_gpt4 = merged_df['base_name'].tolist()
    base_names_combined = base_names_taxo + base_names_gpt4
    
    # Prepare image_paths duplicated for Taxonomist and GPT-4
    image_paths_combined = image_paths + image_paths  # Duplicate image paths for both datasets
    
    # Plot PaCMAP with Thumbnails Only and Connections
    plot_pacmap_with_thumbnails_only_and_connections(
        embeddings_taxo=text_emb_taxo.numpy(),
        embeddings_gpt4=text_emb_gpt4.numpy(),
        labels=labels,
        image_paths=image_paths_combined,
        base_names=base_names_combined,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        category_name=args.category_name,
        thumbnail_size=tuple(args.thumbnail_size),
        thumbnail_offset=tuple(args.thumbnail_offset),
        n_neighbors=args.n_neighbors
    )
    
    # -----------------------------
    # Added: Handle Multiple Models Comparison
    # -----------------------------
    if args.models_csv:
        # Import necessary libraries for the new functionality
        import glob  # Already imported at the top
        
        # -----------------------------
        # 26. Load Multiple Models from CSV
        # -----------------------------
        def load_multiple_models(models_csv_path):
            """
            Load multiple models' descriptions from a CSV file.
            
            Parameters:
            - models_csv_path: Path to the CSV file.
            
            Returns:
            - models_df: DataFrame containing 'base_name', 'model_name', 'description', and 'image_path'.
            """
            if not os.path.exists(models_csv_path):
                logging.error(f"Models CSV file '{models_csv_path}' does not exist.")
                print(f"Error: Models CSV file '{models_csv_path}' does not exist.")
                sys.exit(1)
            
            try:
                models_df = pd.read_csv(models_csv_path)
                logging.info(f"Loaded models data from '{models_csv_path}'")
                print(f"Loaded models data from '{models_csv_path}'")
            except Exception as e:
                logging.error(f"Error loading models CSV file '{models_csv_path}': {e}")
                print(f"Error loading models CSV file '{models_csv_path}': {e}")
                sys.exit(1)
            
            # Check for required columns
            required_columns = {'base_name', 'model_name', 'description', 'image_path'}
            if not required_columns.issubset(models_df.columns):
                logging.error(f"Models CSV must contain the following columns: {required_columns}")
                print(f"Error: Models CSV must contain the following columns: {required_columns}")
                sys.exit(1)
            
            # Standardize 'base_name'
            models_df['base_name'] = models_df['base_name'].str.lower().str.strip()
            
            return models_df
        
        # -----------------------------
        # 27. Compute Embeddings for Multiple Models
        # -----------------------------
        def compute_embeddings_for_multiple_models(models_df, device='cpu', max_tokens=70, stride=35):
            """
            Compute CLIP embeddings for multiple models.
            
            Parameters:
            - models_df: DataFrame containing 'base_name', 'model_name', 'description', and 'image_path'.
            - device: 'cpu' or 'cuda' for GPU acceleration.
            - max_tokens: Maximum number of tokens allowed for each window (<=77).
            - stride: Number of tokens to move the window each step.
            
            Returns:
            - embeddings_dict: Dictionary with model names as keys and DataFrames containing embeddings as values.
            """
            model_names = models_df['model_name'].unique()
            embeddings_dict = {}
            
            for model in model_names:
                logging.info(f"Processing embeddings for model '{model}'")
                print(f"Processing embeddings for model '{model}'")
                
                model_df = models_df[models_df['model_name'] == model].reset_index(drop=True)
                
                image_paths = model_df['image_path'].tolist()
                descriptions = model_df['description'].tolist()
                
                image_emb, text_emb = generate_clip_embeddings(
                    image_paths,
                    descriptions,
                    device=device,
                    max_tokens=max_tokens,
                    stride=stride
                )
                
                # Create a DataFrame for embeddings
                embeddings_df = pd.DataFrame({
                    'base_name': model_df['base_name'],
                    'image_embedding': list(image_emb.numpy()),
                    'text_embedding': list(text_emb.numpy())
                })
                
                embeddings_dict[model] = embeddings_df
                logging.info(f"Embeddings computed for model '{model}'")
                print(f"Embeddings computed for model '{model}'")
            
            return embeddings_dict

        # -----------------------------
        # 28. Compare Embeddings Across Multiple Models
        # -----------------------------
        def compare_embeddings_across_models(embeddings_dict, output_dir):
            """
            Compare embeddings across multiple models by computing cosine similarities.
            
            Parameters:
            - embeddings_dict: Dictionary with model names as keys and DataFrames containing embeddings as values.
            - output_dir: Directory to save the comparison results.
            
            Returns:
            - similarity_matrix: DataFrame containing cosine similarities between models.
            """
            models = list(embeddings_dict.keys())
            comparison_records = []
            
            # Ensure all models have the same 'base_name' entries
            base_names = set.intersection(*(set(df['base_name']) for df in embeddings_dict.values()))
            if not base_names:
                logging.error("No common 'base_name' entries found across models for comparison.")
                print("Error: No common 'base_name' entries found across models for comparison.")
                return pd.DataFrame()
            
            # Iterate over each 'base_name'
            for base in base_names:
                record = {'base_name': base}
                for model in models:
                    df = embeddings_dict[model]
                    base_entry = df[df['base_name'] == base]
                    if not base_entry.empty:
                        # Store text embeddings for comparison
                        text_emb = base_entry['text_embedding'].values[0].reshape(1, -1)
                        record[f'text_embedding_{model}'] = text_emb
                    else:
                        record[f'text_embedding_{model}'] = np.nan
                comparison_records.append(record)
            
            comparison_df = pd.DataFrame(comparison_records)
            comparison_df.set_index('base_name', inplace=True)
            
            # Compute pairwise cosine similarities between models
            similarity_matrix = pd.DataFrame(index=models, columns=models)
            
            for i, model_i in enumerate(models):
                for j, model_j in enumerate(models):
                    if i <= j:
                        # Extract embeddings
                        emb_i = np.vstack(comparison_df[f'text_embedding_{model_i}'].values)
                        emb_j = np.vstack(comparison_df[f'text_embedding_{model_j}'].values)
                        
                        # Compute cosine similarities
                        similarities = cosine_similarity(emb_i, emb_j).diagonal()
                        
                        # Store the average similarity
                        avg_similarity = similarities.mean()
                        similarity_matrix.loc[model_i, model_j] = avg_similarity
                        similarity_matrix.loc[model_j, model_i] = avg_similarity
            
            # Save similarity matrix to CSV
            similarity_csv_path = os.path.join(output_dir, 'embedding_similarity_matrix.csv')
            similarity_matrix.to_csv(similarity_csv_path)
            logging.info(f"Embedding similarity matrix saved to '{similarity_csv_path}'")
            print(f"Embedding similarity matrix saved to '{similarity_csv_path}'")
            
            return similarity_matrix

        # -----------------------------
        # Added: New Functions are already defined above
        # -----------------------------
        
        # -----------------------------
        # 26. Load Multiple Models from CSV
        # -----------------------------
        models_df = load_multiple_models(args.models_csv)
        
        # -----------------------------
        # 27. Compute Embeddings for Multiple Models
        # -----------------------------
        embeddings_dict = compute_embeddings_for_multiple_models(
            models_df,
            device=device,
            max_tokens=70,
            stride=35
        )
        
        # -----------------------------
        # 28. Compare Embeddings Across Multiple Models
        # -----------------------------
        similarity_matrix = compare_embeddings_across_models(
            embeddings_dict,
            output_dir=args.output_dir
        )
        
        if not similarity_matrix.empty:
            # Optionally, visualize the similarity matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Embedding Cosine Similarity Matrix Across Models')
            plt.tight_layout()
            similarity_heatmap_path = os.path.join(args.output_dir, 'embedding_similarity_heatmap.png')
            plt.savefig(similarity_heatmap_path, dpi=300)
            plt.close()
            logging.info(f"Embedding similarity heatmap saved to '{similarity_heatmap_path}'")
            print(f"Embedding similarity heatmap saved to '{similarity_heatmap_path}'")
        
        # Additional visualizations or analyses can be added here as needed

# -----------------------------
# Added: Define Load Multiple Models and Related Functions Above
# -----------------------------

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
