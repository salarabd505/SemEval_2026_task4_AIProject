import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ADVANCED: For embeddings and similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import numpy as np

# =========================
# DATA LOADING & Basic EDA
# =========================

# Load JSONL file
data = []
with open('dev_track_a.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

print(df.head())
print(df.info())
print(df.isnull().sum())

# Basic text length stats
df['anchor_length'] = df['anchor_text'].str.split().str.len()
df['a_length'] = df['text_a'].str.split().str.len()
df['b_length'] = df['text_b'].str.split().str.len()
print(df[['anchor_length', 'a_length', 'b_length']].describe())

# =========================
# PLOTS
# =========================

# 1. Text Length Distribution (Histogram)
plt.hist(df['anchor_length'], bins=30, alpha=0.5, label='Anchor')
plt.hist(df['a_length'], bins=30, alpha=0.5, label='A')
plt.hist(df['b_length'], bins=30, alpha=0.5, label='B')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.legend()
plt.title('Text Length Distribution')
plt.tight_layout()
plt.savefig('plots/text_length_distribution.pdf')
plt.close()

# 2. Boxplot of Text Lengths
data_box = [df['anchor_length'], df['a_length'], df['b_length']]
plt.boxplot(data_box, tick_labels=['Anchor', 'A', 'B'])
plt.ylabel('Token Count')
plt.title('Boxplot of Text Lengths')
plt.tight_layout()
plt.savefig('plots/box_text_lengths.pdf')
plt.close()

# 3. Vocabulary Size
unique_words = set()
for col in ['anchor_text', 'text_a', 'text_b']:
    unique_words.update(' '.join(df[col]).split())
print("Vocabulary size:", len(unique_words))

# =========================
# ADVANCED PLOTS
# =========================

# Create BERT or S-BERT sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
anchor_emb = model.encode(df['anchor_text'].tolist(), convert_to_numpy=True)
a_emb = model.encode(df['text_a'].tolist(), convert_to_numpy=True)
b_emb = model.encode(df['text_b'].tolist(), convert_to_numpy=True)

# --- Cosine similarities ---
cos_sim_a = np.array([cosine_similarity(anchor_emb[i].reshape(1,-1), a_emb[i].reshape(1,-1))[0,0] for i in range(len(df))])
cos_sim_b = np.array([cosine_similarity(anchor_emb[i].reshape(1,-1), b_emb[i].reshape(1,-1))[0,0] for i in range(len(df))])

# 4. Cosine Similarity Distribution Plot
plt.hist(cos_sim_a, bins=30, alpha=0.7, label='Anchor-A')
plt.hist(cos_sim_b, bins=30, alpha=0.7, label='Anchor-B')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Cosine Similarity Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('plots/cosine_similarity_distribution.pdf')
plt.close()

# 5. Similarity Differences Plot
diff = cos_sim_a - cos_sim_b
plt.hist(diff, bins=30, color='purple', alpha=0.7)
plt.xlabel('Cosine Similarity Difference (A minus B)')
plt.ylabel('Frequency')
plt.title('Difference in Cosine Similarity Scores')
plt.tight_layout()
plt.savefig('plots/similarity_difference.pdf')
plt.close()

# 6. Confusion Matrix (needs ground truth and prediction)
pred_labels = (cos_sim_a > cos_sim_b).astype(int)
true_labels = df['text_a_is_closer'].astype(int).values
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B closer', 'A closer'])
disp.plot(cmap='Blues', xticks_rotation='horizontal')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.pdf')
plt.close()

# 7. Class Balance Plot
label_counts = Counter(df['text_a_is_closer'])
plt.bar(['A closer', 'B closer'], [label_counts[1], label_counts[0]], color=['green', 'orange'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Label Distribution (Class Balance)')
plt.tight_layout()
plt.savefig('plots/class_balance.pdf')
plt.close()

# 8. Sentence Length Distribution Plot
def sentence_lengths(text):
    sentences = [sent.strip() for sent in text.replace('?', '.').replace('!', '.').split('.') if sent.strip()]
    return [len(sent.split()) for sent in sentences]

lengths = []
for col in ['anchor_text', 'text_a', 'text_b']:
    lengths += sum(df[col].apply(sentence_lengths).tolist(), [])

plt.hist(lengths, bins=30, color='teal', alpha=0.7)
plt.xlabel('Tokens per Sentence')
plt.ylabel('Frequency')
plt.title('Sentence Length Distribution')
plt.tight_layout()
plt.savefig('plots/sentence_length_distribution.pdf')
plt.close()

# 9. Embedding Space Visualization (t-SNE on all anchor, A, B)
all_embs = np.vstack([anchor_emb, a_emb, b_emb])
labels = (['Anchor']*len(df)) + (['A']*len(df)) + (['B']*len(df))
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
embs_2d = tsne.fit_transform(all_embs)
plt.figure(figsize=(8, 6))
plt.scatter(embs_2d[:len(df), 0], embs_2d[:len(df), 1], label='Anchor', alpha=0.5, s=10)
plt.scatter(embs_2d[len(df):2*len(df), 0], embs_2d[len(df):2*len(df), 1], label='A', alpha=0.5, s=10)
plt.scatter(embs_2d[2*len(df):, 0], embs_2d[2*len(df):, 1], label='B', alpha=0.5, s=10)
plt.legend()
plt.title('t-SNE Embedding Space Visualization')
plt.tight_layout()
plt.savefig('plots/tsne_embedding_space.pdf')
plt.close()

print("All EDA and advanced plots saved to 'plots/' folder.")
