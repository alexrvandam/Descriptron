#!/usr/bin/env python
"""
calculate_rogueV38_modifiedv2.py

This script computes ROUGE scores, builds language synteny graphs (adjective–noun co‐occurrence networks),
and produces visualizations (including ROUGE heatmaps and PaCMAP plots with image thumbnails).

It accepts either:
   - A single TSV file (via --tsv_file) with columns: base_name, description_taxo, image_path, description_gpt4
   - OR separate files via --taxo_path and --gpt4_text_path

It also computes a simple comparison of the language synteny graphs (using Jaccard similarity for nodes and edges).

Usage Example:
  conda run -n gpt4 python calculate_rogueV38_modifiedv2.py \
      --tsv_file '/home/localuser/Downloads/embeddings - embeddings.tsv' \
      --image_dir '/home/localuser/psilidae/Russelliana_forewing_croped_JPEG/' \
      --mask_dir '/home/localuser/psilidae/Russelliana_forewing_croped_JPEG/binary_foreground_mask_out/entire_forewing/' \
      --output_dir '/home/localuser/psilidae/Russelliana_forewing_croped_JPEG/binary_foreground_mask_out/entire_forewing/rogue_out_v38_mod' \
      --category_name entire_forewing
"""

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
from matplotlib_venn import venn2  # For Venn diagrams
from sklearn.neighbors import NearestNeighbors  # For connecting species
from scipy.stats import wilcoxon  # For statistical tests
from matplotlib.patches import Patch  # For legend handles

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    filename='calculate_rogueV38_modifiedv2.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting object dtype arrays on .fillna")
warnings.filterwarnings("ignore", category=FutureWarning, message="Passing `palette` without assigning `hue` is deprecated")

# Initialize the CLIP tokenizer globally
tokenizer = SimpleTokenizer()

# -----------------------------
# 1. Initialize ROUGE Scorer and spaCy
# -----------------------------
def initialize_tools():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    try:
        nlp = spacy.load('en_core_web_sm')
        logging.info("SpaCy model 'en_core_web_sm' loaded successfully.")
    except OSError:
        logging.error("SpaCy model 'en_core_web_sm' not found.")
        print("Error: SpaCy model 'en_core_web_sm' not found.\nPlease install it using: python -m spacy download en_core_web_sm")
        sys.exit(1)
    return scorer, nlp

# -----------------------------
# 2. Define preprocess_text Function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# 3. Load Data from TSV or Separate Files
# -----------------------------
def load_data(tsv_file=None, taxo_path=None, gpt4_text_path=None):
    """
    If tsv_file is provided, load a TSV with columns:
      base_name, description_taxo, image_path, description_gpt4.
    Otherwise, load from separate files.
    """
    if tsv_file:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            required = {'base_name', 'description_taxo', 'image_path', 'description_gpt4'}
            if not required.issubset(df.columns):
                logging.error(f"TSV file must contain columns: {required}")
                print(f"Error: TSV file must contain columns: {required}")
                sys.exit(1)
            df['base_name'] = df['base_name'].str.lower().str.strip()
            taxo_df = df[['base_name', 'description_taxo', 'image_path']].copy()
            gpt4_df = df[['base_name', 'description_gpt4']].copy()
            logging.info(f"Loaded data from TSV file {tsv_file}")
            return taxo_df, gpt4_df
        except Exception as e:
            logging.error(f"Error loading TSV file {tsv_file}: {e}")
            print(f"Error loading TSV file {tsv_file}: {e}")
            sys.exit(1)
    else:
        # Use separate files if TSV is not provided.
        taxo_df = load_taxonomist_data(taxo_path)
        gpt4_df = parse_gpt4_text_file(gpt4_text_path)
        return taxo_df, gpt4_df

# -----------------------------
# 4. Load Taxonomist TSV Data (for separate files)
# -----------------------------
def load_taxonomist_data(taxo_path):
    try:
        df = pd.read_csv(taxo_path, sep='\t')
        logging.info(f"Loaded Taxonomist data from '{taxo_path}'")
    except Exception as e:
        logging.error(f"Error loading Taxonomist data from '{taxo_path}': {e}")
        print(f"Error loading Taxonomist data from '{taxo_path}': {e}")
        sys.exit(1)
    required_columns = {'base_name', 'answer', 'image_path'}
    if not required_columns.issubset(df.columns):
        logging.error(f"Taxonomist data must contain columns: {required_columns}")
        print(f"Error: Taxonomist data must contain columns: {required_columns}")
        sys.exit(1)
    df = df[['base_name', 'answer', 'image_path']].rename(columns={'answer': 'description_taxo'})
    df['base_name'] = df['base_name'].str.lower().str.strip()
    return df

# -----------------------------
# 5. Verify Image Paths
# -----------------------------
def verify_image_paths(taxo_df):
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
# 6. Parse GPT-4 Text File (for separate files)
# -----------------------------
def parse_gpt4_text_file(gpt4_text_path):
    if not os.path.exists(gpt4_text_path):
        logging.error(f"GPT-4 text file '{gpt4_text_path}' does not exist.")
        print(f"Error: GPT-4 text file '{gpt4_text_path}' does not exist.")
        sys.exit(1)
    with open(gpt4_text_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split('---')
    records = []
    failed = 0
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        m = re.search(r"Species Description for\s+\*?([^\*:\n]+)\*?:", sec, re.IGNORECASE)
        if not m:
            logging.warning("Could not extract base_name from a section; skipping.")
            failed += 1
            continue
        base_name = m.group(1).strip().lower()
        dmatch = re.search(r"Description:\s*\n\s*\n(.*?)(\n\s*\n|$)", sec, re.DOTALL | re.IGNORECASE)
        if not dmatch:
            logging.warning(f"Could not extract Description for '{base_name}'; skipping.")
            failed += 1
            continue
        desc = dmatch.group(1).strip()
        if not desc:
            logging.warning(f"No description found for '{base_name}'; skipping.")
            failed += 1
            continue
        records.append({'base_name': base_name, 'description_gpt4': desc})
    df = pd.DataFrame(records)
    logging.info(f"Parsed {len(df)} GPT-4 descriptions; {failed} failed.")
    print(f"Parsed {len(df)} GPT-4 descriptions from '{gpt4_text_path}'")
    if failed > 0:
        print(f"Failed to parse {failed} sections.")
    return df

# -----------------------------
# 7. Compute ROUGE Scores
# -----------------------------
def compute_rouge_scores(taxo_df, gpt4_df, scorer):
    results = []
    merged_df = pd.merge(taxo_df, gpt4_df, on='base_name', how='inner')
    logging.info(f"Merged {len(merged_df)} records based on 'base_name'")
    print(f"Merged {len(merged_df)} records based on 'base_name'")
    for idx, row in merged_df.iterrows():
        base = row['base_name']
        taxo_desc = row['description_taxo']
        gpt4_desc = row['description_gpt4']
        if pd.isna(taxo_desc) or pd.isna(gpt4_desc):
            logging.warning(f"Missing description for '{base}'; skipping.")
            continue
        scores = scorer.score(taxo_desc, gpt4_desc)
        results.append({
            'base_name': base,
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
    return pd.DataFrame(results)

# -----------------------------
# 8. Plot ROUGE Heatmap
# -----------------------------
def plot_rouge_heatmap(rouge_df, output_dir, save=False, save_path=None):
    heatmap_data = rouge_df.set_index('base_name')[['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']]
    plt.figure(figsize=(12, 20))
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
# 9. Extract Adjectives and Nouns and Build Co-occurrence DataFrames
# -----------------------------
def extract_adj_nouns(text, nlp):
    text = preprocess_text(text)
    doc = nlp(text)
    adj = [token.text for token in doc if token.pos_ == 'ADJ']
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return adj, nouns

def build_adj_noun_df(taxo_df, gpt4_df, nlp):
    taxo_records = []
    for idx, row in taxo_df.iterrows():
        base = row['base_name']
        desc = row['description_taxo']
        if pd.isna(desc):
            continue
        adjs, nouns = extract_adj_nouns(desc, nlp)
        taxo_records.append({'base_name': base, 'adjectives': adjs, 'nouns': nouns})
    taxo_adj_df = pd.DataFrame(taxo_records, columns=['base_name', 'adjectives', 'nouns'])
    
    gpt4_records = []
    for idx, row in gpt4_df.iterrows():
        base = row['base_name']
        desc = row['description_gpt4']
        if pd.isna(desc):
            continue
        adjs, nouns = extract_adj_nouns(desc, nlp)
        gpt4_records.append({'base_name': base, 'adjectives': adjs, 'nouns': nouns})
    gpt4_adj_df = pd.DataFrame(gpt4_records, columns=['base_name', 'adjectives', 'nouns'])
    logging.info("Extracted adjectives and nouns from both datasets.")
    print("Extracted adjectives and nouns from both datasets.")
    return taxo_adj_df, gpt4_adj_df

def build_cooccurrence(dataframe):
    cooc = defaultdict(lambda: defaultdict(int))
    for idx, row in dataframe.iterrows():
        for a in row['adjectives']:
            for n in row['nouns']:
                cooc[a][n] += 1
    return cooc

# -----------------------------
# 10. Plot Co-occurrence Network with Enhanced Features
# -----------------------------
def plot_cooccurrence(cooc_dict, title, top_n=30, save=False, save_path=None):
    edges = []
    node_frequency = defaultdict(int)
    for adj, nouns in cooc_dict.items():
        for noun, count in nouns.items():
            edges.append((adj, noun, count))
            node_frequency[adj] += count
            node_frequency[noun] += count
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]
    if not edges:
        logging.warning(f"No edges to plot for '{title}'.")
        print(f"No edges to plot for '{title}'.")
        if save and save_path:
            plt.figure(figsize=(12, 8))
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            logging.info(f"Empty plot saved to '{save_path}'")
            print(f"Empty plot saved to '{save_path}'")
        return
    G = nx.Graph()
    for adj, noun, weight in edges:
        G.add_edge(adj, noun, weight=weight)
    pos = nx.spring_layout(G, k=0.5, seed=42)
    frequencies = [node_frequency[node] for node in G.nodes()]
    node_cmap = plt.get_cmap('YlGnBu')
    norm_node = plt.Normalize(min(frequencies), max(frequencies))
    node_colors = [node_cmap(norm_node(freq)) for freq in frequencies]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_cmap = plt.get_cmap('hot')
    norm_edge = plt.Normalize(min(edge_weights), max(edge_weights))
    edge_colors = [edge_cmap(norm_edge(w)) for w in edge_weights]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('lightgray')
    fig.patch.set_facecolor('lightgray')
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
    sm_node = plt.cm.ScalarMappable(cmap=node_cmap, norm=norm_node)
    sm_node.set_array(frequencies)
    sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm_edge)
    sm_edge.set_array(edge_weights)
    cbar_node = fig.colorbar(sm_node, ax=ax, shrink=0.5, aspect=10)
    cbar_node.set_label('Word Frequency', rotation=270, labelpad=15)
    cbar_edge = fig.colorbar(sm_edge, ax=ax, shrink=0.5, aspect=10)
    cbar_edge.set_label('Co-occurrence Count', rotation=270, labelpad=15)
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
# 11. Plot Venn Diagram Function
# -----------------------------
def plot_venn_diagram(set1, set2, labels, title, save=False, save_path=None):
    plt.figure(figsize=(8,8))
    venn2([set1, set2], set_labels=labels)
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
# 12. Count Word Frequencies
# -----------------------------
def count_words(dataframe, column='adjectives'):
    if column not in dataframe.columns:
        logging.warning(f"Column '{column}' does not exist.")
        print(f"Warning: Column '{column}' does not exist in DataFrame.")
        return Counter()
    all_words = []
    for items in dataframe[column]:
        all_words.extend(items)
    return Counter(all_words)

# -----------------------------
# 13. Plot Top Words with Gradient Colors
# -----------------------------
def plot_top_words(counter, title, top_n=20, save=False, save_path=None):
    if not counter:
        logging.warning(f"No words to plot for '{title}'.")
        print(f"No words to plot for '{title}'.")
        return
    most_common = counter.most_common(top_n)
    words, counts = zip(*most_common)
    plt.figure(figsize=(10,6))
    norm = plt.Normalize(min(counts), max(counts))
    cmap = plt.cm.viridis
    colors = [cmap(norm(count)) for count in counts]
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
# 14. Save Word Frequencies to CSV
# -----------------------------
def save_word_frequencies(taxo_adj_freq, gpt4_adj_freq, taxo_noun_freq, gpt4_noun_freq, output_path='word_frequency_comparison.csv'):
    freq_df = pd.DataFrame({
        'Taxonomist_Adj': pd.Series(taxo_adj_freq),
        'GPT4_Adj': pd.Series(gpt4_adj_freq),
        'Taxonomist_Noun': pd.Series(taxo_noun_freq),
        'GPT4_Noun': pd.Series(gpt4_noun_freq)
    }).fillna(0).infer_objects()
    freq_df.to_csv(output_path, index=True)
    logging.info(f"Word frequency comparison saved to '{output_path}'")
    print(f"Word frequency comparison saved to '{output_path}'")

# -----------------------------
# 15. Generate CLIP Embeddings with Sliding Window
# -----------------------------
def generate_clip_embeddings(image_paths, descriptions, device='cpu', max_tokens=70, stride=35):
    model, preprocess = clip.load("ViT-B/32", device=device)
    images = []
    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
            images.append(image)
        except Exception as e:
            logging.error(f"Error loading image '{img_path}': {e}")
            print(f"Error loading image '{img_path}': {e}")
            images.append(torch.zeros(1, 3, 224, 224))
    images = torch.cat(images).to(device)
    all_windows = []
    window_mapping = []
    for idx, desc in enumerate(descriptions):
        if pd.isna(desc):
            logging.warning(f"Missing description at index {idx}.")
            desc = ""
        windows = split_into_windows(desc, max_tokens=max_tokens, stride=stride)
        all_windows.extend(windows)
        window_mapping.extend([idx] * len(windows))
    try:
        texts = clip.tokenize(all_windows).to(device)
    except RuntimeError as e:
        logging.error(f"Error tokenizing texts: {e}")
        print(f"Error tokenizing texts: {e}")
        sys.exit(1)
    with torch.no_grad():
        text_embeddings = model.encode_text(texts)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    aggregated_text_embeddings = []
    text_embeddings = text_embeddings.cpu().numpy()
    window_mapping = np.array(window_mapping)
    text_embeddings = np.array(text_embeddings)
    for idx in range(len(descriptions)):
        windows_idx = np.where(window_mapping == idx)[0]
        if len(windows_idx) == 0:
            aggregated_emb = np.zeros(text_embeddings.shape[1])
        else:
            aggregated_emb = text_embeddings[windows_idx].mean(axis=0)
        aggregated_text_embeddings.append(aggregated_emb)
    aggregated_text_embeddings = torch.tensor(np.array(aggregated_text_embeddings))
    aggregated_text_embeddings /= aggregated_text_embeddings.norm(dim=-1, keepdim=True)
    return images.cpu(), aggregated_text_embeddings.cpu()

# -----------------------------
# 16. Split into Windows Function
# -----------------------------
def split_into_windows(text, max_tokens=70, stride=35):
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
# 17. Extract Network Metrics
# -----------------------------
def extract_network_metrics(cooc_dict):
    G = nx.Graph()
    for adj, nouns in cooc_dict.items():
        for noun, count in nouns.items():
            G.add_edge(adj, noun, weight=count)
    if G.number_of_nodes() == 0:
        logging.warning("The graph has no nodes. Network metrics set to None.")
        print("Warning: The graph has no nodes. Network metrics set to None.")
        return {'density': None, 'average_clustering': None, 'average_degree': None}
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    degrees = [deg for node, deg in G.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    return {'density': density, 'average_clustering': avg_clustering, 'average_degree': avg_degree}

# -----------------------------
# 18. Compare Network Metrics
# -----------------------------
def compare_network_metrics(metrics_taxo, metrics_gpt4):
    comparison_results = {}
    for key in metrics_taxo[0].keys():
        taxo_values = [m[key] for m in metrics_taxo]
        gpt4_values = [m[key] for m in metrics_gpt4]
        if all(v is not None for v in taxo_values) and all(v is not None for v in gpt4_values):
            stat, p = wilcoxon(taxo_values, gpt4_values)
            comparison_results[key] = {'Wilcoxon_stat': stat, 'p-value': p}
        else:
            comparison_results[key] = {'Wilcoxon_stat': None, 'p-value': None}
            logging.warning(f"Cannot perform Wilcoxon test for '{key}' due to missing data.")
            print(f"Warning: Cannot perform Wilcoxon test for '{key}' due to missing data.")
    return comparison_results

# -----------------------------
# 19. Summarize Network Metrics
# -----------------------------
def summarize_metrics(metrics_list):
    summary = {}
    for key in metrics_list[0].keys():
        valid = [m[key] for m in metrics_list if m[key] is not None]
        summary[key] = {'mean': pd.Series(valid).mean() if valid else None,
                        'std': pd.Series(valid).std() if valid else None}
    return pd.DataFrame(summary).T

# -----------------------------
# 20. Compare Synteny Graphs (New Function)
# -----------------------------
def compare_synteny_graphs(taxo_cooc, gpt4_cooc):
    """
    Compare the Taxonomist and GPT-4 co-occurrence graphs by building NetworkX graphs and computing:
      - Jaccard index for node sets
      - Jaccard index for edge sets
      - Basic network metrics for each graph.
    Returns a dictionary with the comparison metrics.
    """
    def graph_from_cooc(cooc):
        G = nx.Graph()
        for adj, nouns in cooc.items():
            for noun, count in nouns.items():
                G.add_edge(adj, noun, weight=count)
        return G

    G_taxo = graph_from_cooc(taxo_cooc)
    G_gpt4 = graph_from_cooc(gpt4_cooc)
    
    nodes_taxo = set(G_taxo.nodes())
    nodes_gpt4 = set(G_gpt4.nodes())
    jaccard_nodes = len(nodes_taxo & nodes_gpt4) / len(nodes_taxo | nodes_gpt4) if (nodes_taxo | nodes_gpt4) else None

    edges_taxo = set(G_taxo.edges())
    edges_gpt4 = set(G_gpt4.edges())
    jaccard_edges = len(edges_taxo & edges_gpt4) / len(edges_taxo | edges_gpt4) if (edges_taxo | edges_gpt4) else None

    metrics_taxo = extract_network_metrics(taxo_cooc)
    metrics_gpt4 = extract_network_metrics(gpt4_cooc)

    return {
        'jaccard_node_similarity': jaccard_nodes,
        'jaccard_edge_similarity': jaccard_edges,
        'taxonomist_metrics': metrics_taxo,
        'gpt4_metrics': metrics_gpt4
    }

# -----------------------------
# 21. Create Thumbnail With Contour
# -----------------------------
def create_thumbnail(image, mask, thumbnail_size=(80, 80), border_color=None, border_thickness=2):
    binary_mask = (mask > 0).astype(np.uint8)
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(foreground_rgb)
    pil_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
    pil_img.putalpha(pil_mask)
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)
    if border_color:
        thumbnail_np = np.array(pil_img)
        alpha = thumbnail_np[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(thumbnail_np)
        border_color_cv = (border_color[2], border_color[1], border_color[0])
        cv2.drawContours(contour_img, contours, -1, border_color_cv + (255,), thickness=border_thickness)
        contour_alpha = contour_img[:, :, 3] > 0
        thumbnail_np[contour_alpha] = contour_img[contour_alpha]
        pil_img = Image.fromarray(thumbnail_np)
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    offset = ((thumbnail_size[0] - pil_img.size[0]) // 2, (thumbnail_size[1] - pil_img.size[1]) // 2)
    new_img.paste(pil_img, offset, pil_img)
    return np.array(new_img)

# -----------------------------
# 22. Plot PaCMAP with Thumbnails Only and Connect Species
# -----------------------------
def plot_pacmap_with_thumbnails_only_and_connections(embeddings_taxo, embeddings_gpt4, labels, image_paths, base_names, mask_dir, output_dir, category_name, thumbnail_size=(80, 80), thumbnail_offset=(5, 0), n_neighbors=3):
    combined_embeddings = np.vstack((embeddings_taxo, embeddings_gpt4))
    pac = pacmap.PaCMAP(n_components=2, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    embedding_2d = pac.fit_transform(combined_embeddings)
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'Type': labels,
        'image_path': image_paths,
        'base_name': base_names
    })
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f'PaCMAP of Text Embeddings with Thumbnails Only ({category_name})')
    ax.set_xlabel('PaCMAP Component 1')
    ax.set_ylabel('PaCMAP Component 2')
    ax.set_xticks([])
    ax.set_yticks([])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(combined_embeddings)
    distances, indices = nbrs.kneighbors(combined_embeddings)
    for idx, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:
            x_vals = [embedding_2d[idx, 0], embedding_2d[neighbor, 0]]
            y_vals = [embedding_2d[idx, 1], embedding_2d[neighbor, 1]]
            ax.plot(x_vals, y_vals, color='gray', linewidth=0.5, alpha=0.5)
    for idx, row in plot_df.iterrows():
        img_path = row['image_path']
        desc_type = row['Type']
        base = row['base_name']
        if desc_type == 'Taxonomist':
            border_color = (0, 0, 255)
        elif desc_type == 'GPT-4':
            border_color = (255, 0, 0)
        else:
            border_color = None
        base_filename = os.path.basename(img_path)
        mask_pattern = f"binary_mask_{base_filename.split('.')[0]}*.png"
        mask_files = glob.glob(os.path.join(mask_dir, mask_pattern))
        if mask_files:
            mask_path = mask_files[0]
            mask = cv2.imread(mask_path, 0)
            if mask is not None:
                image = cv2.imread(img_path)
                if image is not None:
                    thumbnail = create_thumbnail(image, mask, thumbnail_size=thumbnail_size, border_color=border_color, border_thickness=1)
                    imagebox = OffsetImage(thumbnail, zoom=1.0)
                    ab = AnnotationBbox(imagebox, (row['x'], row['y']),
                                        frameon=False, xybox=thumbnail_offset,
                                        boxcoords="offset points", pad=0.0)
                    ax.add_artist(ab)
    unique_bases = set(base_names)
    for base in unique_bases:
        taxo_idx = plot_df[(plot_df['base_name'] == base) & (plot_df['Type'] == 'Taxonomist')].index
        gpt4_idx = plot_df[(plot_df['base_name'] == base) & (plot_df['Type'] == 'GPT-4')].index
        if len(taxo_idx) == 1 and len(gpt4_idx) == 1:
            taxo_pt = (plot_df.loc[taxo_idx[0], 'x'], plot_df.loc[taxo_idx[0], 'y'])
            gpt4_pt = (plot_df.loc[gpt4_idx[0], 'x'], plot_df.loc[gpt4_idx[0], 'y'])
            ax.plot([taxo_pt[0], gpt4_pt[0]], [taxo_pt[1], gpt4_pt[1]], color='limegreen', linewidth=1.5, alpha=0.8)
        else:
            logging.warning(f"Could not find exactly one Taxonomist and one GPT-4 entry for base_name '{base}'.")
            print(f"Warning: Could not find exactly one Taxonomist and one GPT-4 entry for base_name '{base}'. Skipping connecting line.")
    buffer = max(thumbnail_size) / 100
    x_min, x_max = plot_df['x'].min() - buffer, plot_df['x'].max() + buffer
    y_min, y_max = plot_df['y'].min() - buffer, plot_df['y'].max() + buffer
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Taxonomist'),
        Patch(facecolor='red', edgecolor='red', label='GPT-4'),
        Patch(facecolor='limegreen', edgecolor='limegreen', label='Taxo-GPT4 Connection')
    ]
    ax.legend(handles=legend_elements, title='Description Type', loc='upper right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    pattern = r"\W+"
    filename_part = re.sub(pattern, "_", category_name)
    plot_filename = f'pacmap_embeddings_with_thumbnails_only_connections_{filename_part}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"PaCMAP plot saved to '{plot_path}'")
    print(f"PaCMAP plot saved to: {plot_path}")

# -----------------------------
# 23. Load COCO Masks
# -----------------------------
def load_coco_masks(json_path, category_name='entire_forewing'):
    masks_dict = {}
    if not os.path.exists(json_path):
        logging.error(f"COCO JSON file '{json_path}' does not exist.")
        print(f"Error: COCO JSON file '{json_path}' does not exist.")
        return masks_dict
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            print(f"Error: JSON decoding error: {e}")
            return masks_dict
    image_id_to_filename = {img['id']: img['file_name'] for img in data.get('images', [])}
    logging.info(f"Loaded {len(image_id_to_filename)} images from the JSON.")
    print(f"Loaded {len(image_id_to_filename)} images from the JSON.")
    category_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    logging.info(f"Loaded {len(category_id_to_name)} categories from the JSON.")
    print(f"Loaded {len(category_id_to_name)} categories from the JSON.")
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        cat_name = category_id_to_name.get(cat_id, None)
        if not cat_name:
            logging.warning(f"Annotation ID {ann.get('id')} has unknown category_id {cat_id}.")
            continue
        if cat_name != category_name:
            continue
        image_id = ann.get('image_id')
        filename = image_id_to_filename.get(image_id, None)
        if not filename:
            logging.warning(f"Annotation ID {ann.get('id')} references unknown image_id {image_id}.")
            continue
        mask_pattern = f"binary_mask_{filename.split('.')[0]}*.png"
        mask_files = glob.glob(os.path.join(os.path.dirname(json_path), mask_pattern))
        if mask_files:
            masks_dict[filename] = mask_files[0]
        else:
            logging.warning(f"No mask found for image '{filename}'.")
    logging.info(f"Loaded masks for {len(masks_dict)} images.")
    print(f"Loaded masks for {len(masks_dict)} images.")
    return masks_dict

# -----------------------------
# 24. Save Embeddings Function
# -----------------------------
def save_embeddings(merged_df, output_dir='embeddings'):
    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, 'embeddings.csv')
    merged_df.to_csv(embeddings_path, index=False)
    logging.info(f"Embeddings saved to '{embeddings_path}'")
    print(f"Embeddings saved to '{embeddings_path}'")

# -----------------------------
# 25. Main Execution Function
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate ROUGE scores, build language synteny graphs, and visualize embeddings with thumbnails.")
    # For single TSV input:
    parser.add_argument('--tsv_file', type=str, required=False,
                        help="Path to a TSV file with columns: base_name, description_taxo, image_path, description_gpt4")
    # If TSV is not provided, fallback to separate files:
    parser.add_argument('--taxo_path', type=str, required=False,
                        help="Path to taxonomist TSV file (if no --tsv_file provided)")
    parser.add_argument('--gpt4_text_path', type=str, required=False,
                        help="Path to GPT-4 text file (if no --tsv_file provided)")
    parser.add_argument('--coco_jsonl_path', type=str, required=False, help="Path to COCO JSON file (optional, for masks)")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--category_name', type=str, default='entire_forewing', help='Category name for masks.')
    parser.add_argument('--thumbnail_size', type=int, nargs=2, default=(80, 80), help='Size of thumbnails (width height).')
    parser.add_argument('--thumbnail_offset', type=int, nargs=2, default=(5, 0), help='Offset for thumbnails (x y).')
    parser.add_argument('--n_neighbors', type=int, default=3, help='Number of nearest neighbors to connect each species to.')
    parser.add_argument('--models_csv', type=str, required=False, help='Path to CSV file containing multiple models data for comparison.')
    args = parser.parse_args()
    
    scorer, nlp = initialize_tools()
    
    # Load data using TSV if provided; otherwise, use separate files.
    if args.tsv_file:
        taxo_df, gpt4_df = load_data(tsv_file=args.tsv_file)
    else:
        if not args.taxo_path or not args.gpt4_text_path:
            print("Error: Must provide either --tsv_file or both --taxo_path and --gpt4_text_path.")
            sys.exit(1)
        taxo_df, gpt4_df = load_data(taxo_path=args.taxo_path, gpt4_text_path=args.gpt4_text_path)
    
    taxo_df = verify_image_paths(taxo_df)
    
    print("\nTaxonomist base_names:")
    print(taxo_df['base_name'].tolist())
    print("\nGPT-4 base_names:")
    print(gpt4_df['base_name'].tolist())
    
    overlapping = set(taxo_df['base_name']).intersection(set(gpt4_df['base_name']))
    print(f"\nNumber of overlapping base_names: {len(overlapping)}")
    logging.info(f"Number of overlapping base_names: {len(overlapping)}")
    if len(overlapping) == 0:
        print("Error: No overlapping base_name entries found.")
        sys.exit(1)
    taxo_df = taxo_df[taxo_df['base_name'].isin(overlapping)].reset_index(drop=True)
    gpt4_df = gpt4_df[gpt4_df['base_name'].isin(overlapping)].reset_index(drop=True)
    
    rouge_df = compute_rouge_scores(taxo_df, gpt4_df, scorer)
    rouge_csv = os.path.join(args.output_dir, 'rouge_scores_comparison.csv')
    rouge_df.to_csv(rouge_csv, index=False)
    print(f"ROUGE scores saved to '{rouge_csv}'")
    logging.info(f"ROUGE scores saved to '{rouge_csv}'")
    
    rouge_heatmap_path = os.path.join(args.output_dir, 'rouge_scores_heatmap.png')
    plot_rouge_heatmap(rouge_df, args.output_dir, save=True, save_path=rouge_heatmap_path)
    
    taxo_adj_df, gpt4_adj_df = build_adj_noun_df(taxo_df, gpt4_df, nlp)
    taxo_cooc = build_cooccurrence(taxo_adj_df)
    gpt4_cooc = build_cooccurrence(gpt4_adj_df)
    
    plot_cooccurrence(taxo_cooc, "Taxonomist Description Synteny Map", top_n=50,
                      save=True, save_path=os.path.join(args.output_dir, 'taxonomist_synteny_map.png'))
    plot_cooccurrence(gpt4_cooc, "GPT-4 Description Synteny Map", top_n=50,
                      save=True, save_path=os.path.join(args.output_dir, 'gpt4_synteny_map.png'))
    
    # Compare the synteny graphs and output the comparison stats.
    synteny_comparison = compare_synteny_graphs(taxo_cooc, gpt4_cooc)
    synteny_csv = os.path.join(args.output_dir, 'synteny_graph_comparison.csv')
    pd.DataFrame([synteny_comparison]).to_csv(synteny_csv, index=False)
    print(f"Synteny graph comparison saved to '{synteny_csv}'")
    logging.info(f"Synteny graph comparison saved to '{synteny_csv}'")
    # -----------------------------
    # 8a. Generate Unique Synteny Maps (Unique Adjective-Noun Pairs)
    # -----------------------------
    # First, build the full edge sets from the co-occurrence dictionaries:
    taxo_edges = set((adj, noun) for adj, nouns in taxo_cooc.items() for noun in nouns)
    gpt4_edges = set((adj, noun) for adj, nouns in gpt4_cooc.items() for noun in nouns)
    
    # Compute unique edge sets:
    unique_taxo = taxo_edges - gpt4_edges
    unique_gpt4 = gpt4_edges - taxo_edges
    
    # Build unique co-occurrence dictionaries using the unique edge sets:
    unique_taxo_cooc = defaultdict(lambda: defaultdict(int))
    for adj, noun in unique_taxo:
        unique_taxo_cooc[adj][noun] = taxo_cooc[adj][noun]
    
    unique_gpt4_cooc = defaultdict(lambda: defaultdict(int))
    for adj, noun in unique_gpt4:
        unique_gpt4_cooc[adj][noun] = gpt4_cooc[adj][noun]
    
    # Plot and save the unique synteny maps:
    plot_cooccurrence(
        cooc_dict=unique_taxo_cooc,
        title="Taxonomist Unique Synteny Map",
        top_n=30,  # Displays the top 30 unique pairs (default value)
        save=True,
        save_path=os.path.join(args.output_dir, 'taxonomist_unique_synteny_map.png')
    )
    
    plot_cooccurrence(
        cooc_dict=unique_gpt4_cooc,
        title="GPT-4 Unique Synteny Map",
        top_n=30,  # Displays the top 30 unique pairs (default value)
        save=True,
        save_path=os.path.join(args.output_dir, 'gpt4_unique_synteny_map.png')
    )

    # --- Compare Unique Synteny Graphs ---
    unique_synteny_comparison = compare_synteny_graphs(unique_taxo_cooc, unique_gpt4_cooc)
    uniq_synteny_csv = os.path.join(args.output_dir, 'uniq_synteny_graph_comparison.csv')
    pd.DataFrame([unique_synteny_comparison]).to_csv(uniq_synteny_csv, index=False)
    print(f"Unique synteny graph comparison saved to '{uniq_synteny_csv}'")
    logging.info(f"Unique synteny graph comparison saved to '{uniq_synteny_csv}'")

    
    taxo_adj_freq = count_words(taxo_adj_df, 'adjectives')
    gpt4_adj_freq = count_words(gpt4_adj_df, 'adjectives')
    taxo_noun_freq = count_words(taxo_adj_df, 'nouns')
    gpt4_noun_freq = count_words(gpt4_adj_df, 'nouns')
    
    save_word_frequencies(taxo_adj_freq, gpt4_adj_freq, taxo_noun_freq, gpt4_noun_freq,
                          output_path=os.path.join(args.output_dir, 'word_frequency_comparison.csv'))
    
    plot_top_words(taxo_adj_freq, "Top 20 Adjectives - Taxonomist", top_n=20,
                   save=True, save_path=os.path.join(args.output_dir, 'top_20_adjectives_taxonomist.png'))
    plot_top_words(gpt4_adj_freq, "Top 20 Adjectives - GPT-4", top_n=20,
                   save=True, save_path=os.path.join(args.output_dir, 'top_20_adjectives_gpt4.png'))
    plot_top_words(taxo_noun_freq, "Top 20 Nouns - Taxonomist", top_n=20,
                   save=True, save_path=os.path.join(args.output_dir, 'top_20_nouns_taxonomist.png'))
    plot_top_words(gpt4_noun_freq, "Top 20 Nouns - GPT-4", top_n=20,
                   save=True, save_path=os.path.join(args.output_dir, 'top_20_nouns_gpt4.png'))
    
    taxo_nouns = set(taxo_noun_freq.keys())
    gpt4_nouns = set(gpt4_noun_freq.keys())
    venn_nouns_path = os.path.join(args.output_dir, 'noun_overlap_venn.png')
    plot_venn_diagram(taxo_nouns, gpt4_nouns, ('Taxonomist Nouns', 'GPT-4 Nouns'),
                      'Noun Overlap Between Taxonomist and GPT-4', save=True, save_path=venn_nouns_path)
    
    taxo_adjs = set(taxo_adj_freq.keys())
    gpt4_adjs = set(gpt4_adj_freq.keys())
    venn_adjs_path = os.path.join(args.output_dir, 'adjective_overlap_venn.png')
    plot_venn_diagram(taxo_adjs, gpt4_adjs, ('Taxonomist Adjectives', 'GPT-4 Adjectives'),
                      'Adjective Overlap Between Taxonomist and GPT-4', save=True, save_path=venn_adjs_path)
    
    metrics_taxo = [extract_network_metrics(taxo_cooc)]
    metrics_gpt4 = [extract_network_metrics(gpt4_cooc)]
    summary_taxo = summarize_metrics(metrics_taxo)
    summary_gpt4 = summarize_metrics(metrics_gpt4)
    print("\nTaxonomist Metrics Summary:")
    print(summary_taxo)
    print("\nGPT-4 Metrics Summary:")
    print(summary_gpt4)
    logging.info("Taxonomist Metrics Summary:\n" + str(summary_taxo))
    logging.info("GPT-4 Metrics Summary:\n" + str(summary_gpt4))
    comp_results = compare_network_metrics(metrics_taxo, metrics_gpt4)
    comp_df = pd.DataFrame(comp_results).T
    comp_csv = os.path.join(args.output_dir, 'network_metrics_comparison.csv')
    comp_df.to_csv(comp_csv)
    logging.info("Network metrics comparison saved to 'network_metrics_comparison.csv'")
    print("Network metrics comparison saved to 'network_metrics_comparison.csv'")
    print("\nWilcoxon Signed-Rank Test Results:")
    print(comp_df)
    
    merged_emb_df = pd.merge(taxo_df, gpt4_df, on='base_name', how='inner')
    if len(merged_emb_df) == 0:
        logging.error("No records to process for embeddings since merged_df is empty.")
        print("Error: No records to process for embeddings since merged_df is empty.")
        sys.exit(1)
    image_paths = merged_emb_df['image_path'].tolist()
    descriptions_taxo = merged_emb_df['description_taxo'].tolist()
    descriptions_gpt4 = merged_emb_df['description_gpt4'].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")
    image_emb_taxo, text_emb_taxo = generate_clip_embeddings(image_paths, descriptions_taxo,
                                                              device=device, max_tokens=70, stride=35)
    image_emb_gpt4, text_emb_gpt4 = generate_clip_embeddings(image_paths, descriptions_gpt4,
                                                              device=device, max_tokens=70, stride=35)
    merged_emb_df['image_embedding_taxo'] = list(image_emb_taxo.numpy())
    merged_emb_df['text_embedding_taxo'] = list(text_emb_taxo.numpy())
    merged_emb_df['image_embedding_gpt4'] = list(image_emb_gpt4.numpy())
    merged_emb_df['text_embedding_gpt4'] = list(text_emb_gpt4.numpy())
    save_embeddings(merged_emb_df, output_dir=os.path.join(args.output_dir, 'embeddings'))
    sim = cosine_similarity(np.vstack(merged_emb_df['text_embedding_taxo'].values),
                            np.vstack(merged_emb_df['text_embedding_gpt4'].values)).diagonal()
    merged_emb_df['cosine_similarity'] = sim
    cosine_csv = os.path.join(args.output_dir, 'cosine_similarities.csv')
    merged_emb_df[['base_name', 'cosine_similarity']].to_csv(cosine_csv, index=False)
    logging.info(f"Cosine similarities saved to '{cosine_csv}'")
    print(f"Cosine similarities saved to '{cosine_csv}'")
    plt.figure(figsize=(8,6))
    sns.histplot(merged_emb_df['cosine_similarity'], bins=10, kde=True, color='blue')
    plt.title('Distribution of Cosine Similarities Between Taxonomist and GPT-4 Descriptions')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    cosine_plot = os.path.join(args.output_dir, 'cosine_similarity_distribution.png')
    plt.savefig(cosine_plot, dpi=300)
    plt.close()
    logging.info(f"Cosine similarity distribution plot saved to '{cosine_plot}'")
    print(f"Cosine similarity distribution plot saved to '{cosine_plot}'")
    labels = ['Taxonomist'] * len(text_emb_taxo) + ['GPT-4'] * len(text_emb_gpt4)
    base_names_taxo = merged_emb_df['base_name'].tolist()
    base_names_combined = base_names_taxo + base_names_taxo
    image_paths_combined = image_paths + image_paths
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
    
    if args.models_csv:
        import glob
        def load_multiple_models(models_csv_path):
            if not os.path.exists(models_csv_path):
                logging.error(f"Models CSV file '{models_csv_path}' does not exist.")
                print(f"Error: Models CSV file '{models_csv_path}' does not exist.")
                sys.exit(1)
            try:
                df = pd.read_csv(models_csv_path)
                logging.info(f"Loaded models data from '{models_csv_path}'")
                print(f"Loaded models data from '{models_csv_path}'")
            except Exception as e:
                logging.error(f"Error loading models CSV file '{models_csv_path}': {e}")
                print(f"Error loading models CSV file '{models_csv_path}': {e}")
                sys.exit(1)
            required_columns = {'base_name', 'model_name', 'description', 'image_path'}
            if not required_columns.issubset(df.columns):
                logging.error(f"Models CSV must contain columns: {required_columns}")
                print(f"Error: Models CSV must contain columns: {required_columns}")
                sys.exit(1)
            df['base_name'] = df['base_name'].str.lower().str.strip()
            return df
        
        def compute_embeddings_for_multiple_models(models_df, device='cpu', max_tokens=70, stride=35):
            model_names = models_df['model_name'].unique()
            embeddings_dict = {}
            for model in model_names:
                logging.info(f"Processing embeddings for model '{model}'")
                print(f"Processing embeddings for model '{model}'")
                model_df = models_df[models_df['model_name'] == model].reset_index(drop=True)
                image_paths = model_df['image_path'].tolist()
                descriptions = model_df['description'].tolist()
                image_emb, text_emb = generate_clip_embeddings(image_paths, descriptions,
                                                                device=device,
                                                                max_tokens=max_tokens,
                                                                stride=stride)
                embeddings_df = pd.DataFrame({
                    'base_name': model_df['base_name'],
                    'image_embedding': list(image_emb.numpy()),
                    'text_embedding': list(text_emb.numpy())
                })
                embeddings_dict[model] = embeddings_df
                logging.info(f"Embeddings computed for model '{model}'")
                print(f"Embeddings computed for model '{model}'")
            return embeddings_dict
        
        def compare_embeddings_across_models(embeddings_dict, output_dir):
            models = list(embeddings_dict.keys())
            comparison_records = []
            base_names = set.intersection(*(set(df['base_name']) for df in embeddings_dict.values()))
            if not base_names:
                logging.error("No common 'base_name' entries found across models for comparison.")
                print("Error: No common 'base_name' entries found across models for comparison.")
                return pd.DataFrame()
            for base in base_names:
                record = {'base_name': base}
                for model in models:
                    df = embeddings_dict[model]
                    base_entry = df[df['base_name'] == base]
                    if not base_entry.empty:
                        text_emb = base_entry['text_embedding'].values[0].reshape(1, -1)
                        record[f'text_embedding_{model}'] = text_emb
                    else:
                        record[f'text_embedding_{model}'] = np.nan
                comparison_records.append(record)
            comparison_df = pd.DataFrame(comparison_records)
            comparison_df.set_index('base_name', inplace=True)
            similarity_matrix = pd.DataFrame(index=models, columns=models)
            for i, model_i in enumerate(models):
                for j, model_j in enumerate(models):
                    if i <= j:
                        emb_i = np.vstack(comparison_df[f'text_embedding_{model_i}'].values)
                        emb_j = np.vstack(comparison_df[f'text_embedding_{model_j}'].values)
                        sims = cosine_similarity(emb_i, emb_j).diagonal()
                        avg_sim = sims.mean()
                        similarity_matrix.loc[model_i, model_j] = avg_sim
                        similarity_matrix.loc[model_j, model_i] = avg_sim
            similarity_csv_path = os.path.join(output_dir, 'embedding_similarity_matrix.csv')
            similarity_matrix.to_csv(similarity_csv_path)
            logging.info(f"Embedding similarity matrix saved to '{similarity_csv_path}'")
            print(f"Embedding similarity matrix saved to '{similarity_csv_path}'")
            return similarity_matrix

        models_df = load_multiple_models(args.models_csv)
        embeddings_dict = compute_embeddings_for_multiple_models(models_df, device=device, max_tokens=70, stride=35)
        sim_matrix = compare_embeddings_across_models(embeddings_dict, output_dir=args.output_dir)
        if not sim_matrix.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(sim_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Embedding Cosine Similarity Matrix Across Models')
            plt.tight_layout()
            sim_heatmap_path = os.path.join(args.output_dir, 'embedding_similarity_heatmap.png')
            plt.savefig(sim_heatmap_path, dpi=300)
            plt.close()
            logging.info(f"Embedding similarity heatmap saved to '{sim_heatmap_path}'")
            print(f"Embedding similarity heatmap saved to '{sim_heatmap_path}'")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
