from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from io import BytesIO
import subprocess


_orig_show = plt.show


def show(*args, **kwargs):
    # grab the current figure
    fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    subprocess.run(['kitty', '+kitten', 'icat'], input=buf.read())
    plt.close(fig)


plt.show = show
# ================================
# 1. Define Comments and True Sentiments
# ================================
positive_comments = [
    "I love the new design of your website!",
    "The updated interface is fantastic and easy to navigate.",
    "The added features make the experience much better.",
    "Great job on improving the speed with the latest update!",
    "This update is the best I've seen in a while!",
    "This turned out even better than I expected.",
    "Really impressive work – keep it up!",
    "You’ve done an amazing job here.",
    "I love how this was handled.",
    "This is exactly what I was hoping for.",
    "The attention to detail is fantastic.",
    "Everything feels so smooth and polished.",
    "It’s clear you put a lot of thought into this.",
    "This is genuinely inspiring.",
    "You've captured the mood perfectly.",
    "I can’t stop looking at this – it’s so good.",
    "Every part of this just works.",
    "This is incredibly well done.",
    "It’s refreshing to see something like this.",
    "Excellent execution overall.",
    "You nailed the vibe completely."
]

negative_comments = [
    "I dislike the new layout, it's confusing and cluttered.",
    "The website frequently crashes and is very slow now.",
    "Customer service has been unresponsive after the update.",
    "I'm disappointed with the recent changes.",
    "This missed the mark for me.",
    "It feels rushed.",
    "This could really use more work.",
    "Not a fan of how this was handled.",
    "Something about this doesn’t sit right.",
    "It’s kind of all over the place.",
    "This feels unfinished.",
    "I was hoping for more.",
    "The execution is lacking.",
    "This doesn’t quite make sense.",
    "It didn’t resonate with me at all.",
    "Honestly, it’s disappointing.",
    "This is hard to follow.",
    "There’s too much going on.",
    "It feels like a step back.",
    "The idea is there, but it’s poorly done.",
    "I don’t think this works.",
    "It’s confusing and not in a good way.",
    "Feels uninspired.",
    "Unfortunately, this just isn’t good."
]

neutral_comments = [
    "I need help understanding how to use the new functionality.",
    "The changes are noticeable but overall, it is as expected.",
    "It's a standard update without any major surprises.",
    "This is interesting.",
    "Not sure how I feel about this yet.",
    "It works, I guess.",
    "This is different.",
    "It’s a solid attempt.",
    "I can see where you’re going with this.",
    "It’s not bad, not amazing either.",
    "Just okay for me.",
    "I don’t have strong feelings about it.",
    "Kind of middle-of-the-road.",
    "It’s functional.",
    "There’s potential here.",
    "It’s serviceable.",
    "Doesn’t stand out much, but it’s fine.",
    "This could go either way depending on context.",
    "It does what it’s supposed to do.",
    "It’s… there.",
    "It’s acceptable.",
    "Neither good nor bad, just is.",
    "I guess it’s fine as a starting point."
]

# Combine all comments and assign true sentiment labels.
comments = positive_comments + negative_comments + neutral_comments
true_sentiments = (["positive"] * len(positive_comments) +
                   ["negative"] * len(negative_comments) +
                   ["neutral"] * len(neutral_comments))


# 1. Clean & embed
def clean(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)         # strip URLs
    text = re.sub(r'\s+', ' ', text).strip()
    return text


comments_clean = [clean(c) for c in comments]
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(comments_clean, normalize_embeddings=True)

# 2A. KMeans with silhouette-based k
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    sil_scores.append((k, silhouette_score(embeddings, km.labels_)))
best_k = max(sil_scores, key=lambda x: x[1])[0]

kmeans = KMeans(n_clusters=best_k, random_state=42).fit(embeddings)
km_labels = kmeans.labels_
km_centers = kmeans.cluster_centers_      # ← correct instance attribute

# 2B. HDBSCAN with smaller clusters
hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
hd_labels = hdb.fit_predict(embeddings)

# 3. Compare label distributions
df = pd.DataFrame({
    'comment': comments,
    'true_sentiment': true_sentiments,
    'clean': comments_clean, 
    'km_cluster': km_labels,
    'hdb_cluster': hd_labels
})
print("KMeans distribution:\n", df['km_cluster'].value_counts(normalize=True) * 100)
print("HDBSCAN distribution:\n", df['hdb_cluster'].value_counts(normalize=True) * 100)

# 4. Example: get representative comment for each KMeans cluster
for i in range(best_k):
    idxs = np.where(km_labels == i)[0]
    dists = np.linalg.norm(embeddings[idxs] - km_centers[i], axis=1)
    rep = comments[idxs[np.argmin(dists)]]
    print(f"Cluster {i} rep:", rep)


from keybert import KeyBERT
kw_model = KeyBERT(model)

top_keywords = {}
for i in range(best_k):
    # use the 'km_cluster' column, and the 'clean' column
    docs = df[df['km_cluster'] == i]['clean'].tolist()
    doc = " ".join(docs)
    keywords = kw_model.extract_keywords(doc, top_n=5, stop_words='english')
    top_keywords[i] = [kw for kw, score in keywords]
    print(f"Cluster {i} keywords:", top_keywords[i])


pca = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(8, 6))
for cl in sorted(set(km_labels)):
    mask = km_labels == cl
    plt.scatter(pca[mask, 0], pca[mask, 1], label=f"Cl {cl}", alpha=0.7)
plt.legend(); plt.show()

# 5. PCA scatter & save to PNG
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Project down to 2D
pca_coords = PCA(n_components=2).fit_transform(embeddings)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
for cl in sorted(set(km_labels)):
    mask = km_labels == cl
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               label=f"Cl {cl}", alpha=0.7)
ax.legend()
ax.set_title("KMeans Clusters (PCA projection)")

# Save as high-res PNG
fig.savefig('clusters.png',
            format='png',
            dpi=100,
            bbox_inches='tight')

plt.show()

# 1. Silhouette Score vs. k (for your KMeans)
# sil_scores is a list of (k, score) tuples you computed
ks, scores = zip(*sil_scores)

plt.figure(figsize=(6, 4))
plt.plot(ks, scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("KMeans: Silhouette Score by k")
plt.grid(True)
plt.savefig("silhouette_scores.png", dpi=300, bbox_inches="tight")
plt.show()


# 3. Cluster-size distribution for HDBSCAN
counts = df['hdb_cluster'].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("HDBSCAN Cluster")
plt.ylabel("Number of comments")
plt.title("HDBSCAN Cluster Sizes")
plt.savefig("hdbscan_cluster_sizes.png", dpi=200, bbox_inches="tight")
plt.show()

# 4. True-sentiment breakdown within each KMeans cluster
# pivot to get counts per (cluster, sentiment)
sentiment_counts = df.groupby(['km_cluster', 'true_sentiment'])\
                     .size()\
                     .unstack(fill_value=0)

sentiment_counts.plot(kind='bar', figsize=(8, 5))
plt.xlabel("KMeans Cluster")
plt.ylabel("Comment count")
plt.title("True Sentiment Distribution per KMeans Cluster")
plt.legend(title="Sentiment")
plt.savefig("km_sentiment_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# 1. Prepare a DataFrame with your PCA coords + metadata:
pca_coords = PCA(n_components=2).fit_transform(embeddings)
df_plot = pd.DataFrame({
    'x': pca_coords[:, 0],
    'y': pca_coords[:, 1],
    'comment': comments,
    'km_cluster': km_labels.astype(str),   # cast to str so Plotly treats it as categorical
})

# 2. Create the interactive scatter:
fig = px.scatter(
    df_plot,
    x='x', y='y',
    color='km_cluster',
    hover_data=['comment'],      # what shows up on hover
    title="KMeans Clusters (interactive PCA)",
)

# 3a. In Jupyter:  
fig.show()

# 3b. Standalone HTML:
fig.write_html("interactive_clusters.html", auto_open=True)

