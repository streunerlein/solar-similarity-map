
import numpy as np
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import os, sys
# from adjustText import adjust_text

from visualize_solar import solar_corr
p = os.path.dirname(sys.argv[0]) + '/'
doc_embed = np.loadtxt(p + "doc_embedd.txt")
candidates_embed = np.loadtxt(p + "candidate_embedd.txt")

f = open(p + "candidate_labels.txt")
candidates_labels = [a.strip() for a in f.readlines()]
f.close()

candidates_selected = list()
f = open(p + "candidate_selected1.0.txt")
candidates_selected.append([a.strip() for a in f.readlines()])
f.close()
f = open(p + "candidate_selected0.65.txt")
candidates_selected.append([a.strip() for a in f.readlines()])
f.close()

candidates_embed = np.append(candidates_embed, np.array([doc_embed]), axis=0)
# candidates_embed = normalize(candidates_embed)

similarities = cosine_similarity(candidates_embed) #, np.repeat(doc_embed.reshape(1, -1), len(candidates_embed), axis=0))

doc_sim = cosine_similarity(candidates_embed, doc_embed.reshape(1, -1)) #, np.repeat(doc_embed.reshape(1, -1), len(candidates_embed), axis=0))

# similarities = pairwise_distances(candidates_embed, candidates_embed, metric='cosine')
# p.fill_diagonal(similarities, 1)

# sim_between_norm = similarities/np.nanmax(similarities, axis=0)
# sim_between_norm = \
#      0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)

def plotSolar():
  labels = candidates_labels
  labels.append("DOCUMENT")
  doc_sim_t = doc_sim > 0

  candidates_to_embed = list(map(lambda x: x[1], filter(lambda x: doc_sim_t[x[0]], zip(range(0, len(candidates_embed)), candidates_embed))))
  labels = list(map(lambda x: x[1], filter(lambda x: doc_sim_t[x[0]], zip(range(0, len(labels)), labels))))

  similarities = cosine_similarity(np.array(candidates_to_embed)) #, np.repeat(doc_embed.reshape(1, -1), len(candidates_embed), axis=0))

  fig = plt.figure(constrained_layout=True, figsize=(15, 9.5))
  gs = fig.add_gridspec(2, 6)
  ax_big = fig.add_subplot(gs[:, :4])
  ax_sm1 = fig.add_subplot(gs[0, 4:])
  ax_sm2 = fig.add_subplot(gs[1, 4:])

  ax_big.set_title(r'selection before ranking, with labels', loc='center', fontsize=12, color='dimgray')
  ax_sm1.set_title(r'ranked with $\beta=1.0$', loc='center', fontsize=12, color='dimgray')
  ax_sm2.set_title(r'ranked with $\beta=0.65$', loc='center', fontsize=12, color='dimgray')

  solar_corr(similarities, labels, "DOCUMENT", ax=ax_big, calc_corr=False, selected=[])
  solar_corr(similarities, labels, "DOCUMENT", ax=ax_sm1, moon_orbit=0.4, base_circle_size=40, calc_corr=False, selected=candidates_selected[0], show_labels=False)
  solar_corr(similarities, labels, "DOCUMENT", ax=ax_sm2, moon_orbit=0.4, base_circle_size=40, calc_corr=False, selected=candidates_selected[1], show_labels=False)

  plt.savefig('embed-out.eps', format='eps')
  plt.savefig('embed-out-raster.png', format='png')

  plt.show()

# plotSolar()

def plotHeatmap():
  fig = plt.figure(figsize=(12, 12))
  ax = fig.add_subplot(111)
  im = ax.imshow(similarities)

  candidates_labels.append("DOCUMENT")
  # We want to show all ticks...
  ax.set_yticks(np.arange(len(candidates_labels)))
  # ... and label them with the respective list entries
  ax.set_yticklabels(candidates_labels)
  ax.tick_params(axis="y", labelsize=6)
  
  # We want to show all ticks...
  ax.set_xticks(np.arange(len(candidates_labels)))
  # ... and label them with the respective list entries
  ax.set_xticklabels(candidates_labels)
  ax.tick_params(axis="x", labelsize=6)

  ax.tick_params(top=True, bottom=False,
            labeltop=True, labelbottom=False)

  plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

  fig.tight_layout()
  plt.show()

def plotScatter():
  # similarities = doc_sim
  tsne = MDS(n_components=2, random_state=0, dissimilarity='precomputed')
  np.set_printoptions(suppress=True)
  Y = tsne.fit_transform(1 - similarities)

  x_coords = Y[:-1, 0]
  y_coords = Y[:-1, 1]
  # z_coords = Y[:-1, 2]

  fig = plt.figure(figsize=(10, 10))
  # ax = fig.add_subplot(111, projection='3d')
  ax = fig.add_subplot(111)

  # display scatter plot
  candidate_data = zip(candidates_labels, x_coords, y_coords)
  selected_candidate_data = list(filter(lambda x: x[0] in candidates_selected[0], candidate_data))

  candidate_data = zip(candidates_labels, x_coords, y_coords)
  unselected_candidate_data = list(filter(lambda x: x[0] not in candidates_selected[0], candidate_data))

  doc_coords = Y[-1]

  candidate_points = ax.scatter([c[1] for c in unselected_candidate_data], [c[2] for c in unselected_candidate_data], marker='o')
  selected_candidate_points = ax.scatter([c[1] for c in selected_candidate_data], [c[2] for c in selected_candidate_data], marker='o')
  doc_point = ax.scatter([doc_coords[0]], [doc_coords[1]], marker='o')

  # texts = [ax.text(x, y, z, label + (" (" + str(doc_sim[ix][0]) + ")" if doc_sim[ix][0] > 0.46 else ''), ha='left', va='bottom') for ix, label, x, y, z in zip(range(0, len(candidates_labels)), candidates_labels, x_coords, y_coords, z_coords)]

  # plt.annotate("Whole DOCUMENT", xy=(Y[-1]), xytext=(0, 0), textcoords='offset points')

  plt.legend((candidate_points, selected_candidate_points, doc_point),
            ('Candidate', 'Selected Candidates', 'Document'),
            scatterpoints=1,
            loc='upper right',
            ncol=3,
            fontsize=8)

  ax.set_xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
  ax.set_ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
  # ax.set_zlim(z_coords.min()+0.00005, z_coords.max()+0.00005)
  # plt.xlim(Y[-1][0], Y[-1][0] + 100)
  # plt.ylim(Y[-1][1], Y[-1][1] + 100)
  # adjust_text(texts, va="bottom", ha="left", arrowprops=dict(arrowstyle='->', color='red'))
  plt.show()

plotScatter()