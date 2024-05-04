import numpy as np
import torch
import clip
import matplotlib.pyplot as plt


INSTANCE_COLORS = {
0:  (165.0, 80.0, 115.0),
1: (254., 97., 0.), #orange
2: (120., 94., 240.), #purple 
3: (100., 143., 255.), #blue
4: (220., 38., 127.), #pink
5: (255., 176., 0.), #yellow
6: (100., 143., 255.), 
7: (160.0, 50.0, 50.0), 
8:  (129.0, 0.0, 50.0), 
9:  (255., 176., 0.), 
10: (192.0, 100.0, 119.0), 
11: (149.0, 192.0, 228.0), 
12: (14.0, 0.0, 120.0), 
13: (90., 64., 210.), 
14: (152.0, 200.0, 156.0),
15: (129.0, 103.0, 106.0), 
16: (100.0, 160.0, 100.0),  #
17: (70.0, 70.0, 140.0), 
18: (160.0, 20.0, 60.0), 
19: (20., 130., 20.), 
20: (140.0, 30.0, 60.0),
21:  (20.0, 20.0, 120.0), 
22:  (243.0, 115.0, 68.0), 
23:  (120.0, 162.0, 227.0), 
24:  (100.0, 78.0, 142.0), 
25:  (152.0, 95.0, 163.0), 
26:  (160.0, 20.0, 60.0), 
27:  (100.0, 143.0, 255.0), 
28: (255., 204., 153.),
29: (50., 100., 0.),
}


class QuerySimilarityComputation():
    """Compute similarity between a query and the clip features of 3d objects"""

    def __init__(self,):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed)

        sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
        return sentence_embedding_normalized.squeeze().numpy()
 
    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb

        return scores
    
    def get_colors_for_similarity(self, mask_features, masks, text_query):
        per_mask_scores = self.compute_similarity_scores(mask_features, text_query)

        # colorize the mesh based on the openmask3d per mask scores
        non_zero_points = per_mask_scores > 0.01 #>0.2 #!=0 #>0.01 #!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        # normalize the scores between 0 and 1
        openmask3d_per_mask_scores_rescaled[non_zero_points] = (per_mask_scores[non_zero_points]-per_mask_scores[non_zero_points].min())/(per_mask_scores[non_zero_points].max()-per_mask_scores[non_zero_points].min())

        new_colors = np.zeros((masks.shape[1], 3)) + (0.77, 0.77, 0.77) 
        
        for mask_idx, mask in enumerate(masks[::-1, :]):
            # get color from matplotlib colormap
            new_colors[mask>0.5, :] = plt.cm.jet(openmask3d_per_mask_scores_rescaled[len(masks)-mask_idx-1])[:3]
        
        return new_colors
    
    def get_colors_for_retrieved_instances(self, mask_features, masks, text_query, orig_colors=None, k=None, threshold=None):
        assert k!=None or threshold!=None
        per_mask_scores = self.compute_similarity_scores(mask_features, text_query)
        k = min(k, len(per_mask_scores))

        # colorize the mesh based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0 #>0.2 #!=0 #>0.01 #!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        openmask3d_per_mask_scores_rescaled[non_zero_points] = (per_mask_scores[non_zero_points]-per_mask_scores[non_zero_points].min())/(per_mask_scores[non_zero_points].max()-per_mask_scores[non_zero_points].min())

        if orig_colors is None:
            new_colors = np.ones((masks.shape[1], 3))*0 + (0.77, 0.77, 0.77) #np.asarray(scene_mesh_instseg.vertex_colors)*0.5#
        else:
            new_colors = orig_colors.copy()

        descending_scores_indices = np.argsort(per_mask_scores, axis=0)[::-1]
        descending_scores = per_mask_scores[descending_scores_indices]

        top_k_indices = descending_scores_indices[:k]
        top_k_masks = masks[top_k_indices, :]
        top_k_scores = descending_scores[:k]

        #pdb.set_trace()
        for mask_id, mask in enumerate(top_k_masks):
            new_colors[mask>0.5, :] = [el/255.0 for el in INSTANCE_COLORS[mask_id+1]]

        return new_colors