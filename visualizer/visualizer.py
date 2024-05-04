import os

import clip
import numpy as np
import open3d as o3d
import torch

from similarity import QuerySimilarityComputation


# define the paths
MASK_ROOT = "/teamspace/studios/this_studio/openmask3d/output/2024-05-03-13-20-12-experiment"
OPENMASK3D_FEATURE_ROOT = "/teamspace/studios/this_studio/openmask3d/output/2024-05-03-13-20-12-experiment"
SCENE_PCD_ROOT = "/teamspace/studios/openmask3d/openmask3d/resources/420683/42445132"

# let's start with defining some examples
path_masks = os.path.join(MASK_ROOT, '42445132_masks.pt')
path_features = os.path.join(OPENMASK3D_FEATURE_ROOT, '42445132_openmask3d_features.npy')
path_pcd = os.path.join(SCENE_PCD_ROOT, '42445132.ply')

# load the data
masks = torch.load(path_masks)
masks = masks.T
features = np.load(path_features)
pcd = o3d.io.read_point_cloud(path_pcd)

# define the query
query = 'table'

# compute the colors for the pt cloud visualization
similarity_model = QuerySimilarityComputation()
colors = similarity_model.get_colors_for_similarity(features, masks, query)

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])