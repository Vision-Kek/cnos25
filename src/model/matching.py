import hydra
import torch
import numpy as np
import logging

cossim=torch.nn.CosineSimilarity(dim=-1)

def classify_proposals_through_matching(object_proposal_class_embeddings, ref_samples, matching_strategy='top_5'):
    # proposals: (N, d_emb)
    assert object_proposal_class_embeddings.ndim==2, object_proposal_class_embeddings.shape
    N, d = object_proposal_class_embeddings.shape

    # ref_samples: (n_classes, n_samples, d_emb) = (n, k, d), supported (dict,tuple,list,tensor)
    if isinstance(ref_samples, dict):
        ref_samples = tuple(ref_samples.values())
    elif isinstance(ref_samples, list) or isinstance(ref_samples, tuple):
        ref_samples = torch.stack(ref_samples).to(object_proposal_class_embeddings.device)
    assert ref_samples.ndim == 3, ref_samples.shape
    assert ref_samples.shape[-1] == d, ref_samples.shape
    n, k, d = ref_samples.shape
    assert N*n*k*d < 1e9, (N,n,k,d) # memory safety: less than 1G

    if matching_strategy=='prototype':
        ref_samples = ref_samples.mean(dim=1, keepdim=True)
        k = 1
    # each proposal gets n_classes x n_samples scores
    scores = cossim(object_proposal_class_embeddings[:,None,None,:], ref_samples[None,:,:,:])
    assert scores.shape == (N,n,k), scores.shape

    if matching_strategy=='top_5':
        # the CNOS default matching strategy is max_top_5 over k-dim
        score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
        score_per_proposal_and_object = torch.mean(score_per_proposal_and_object, dim=-1)
    elif matching_strategy=='mean' or matching_strategy=='prototype':
        score_per_proposal_and_object = scores.mean(dim=-1)
    elif matching_strategy == 'max':
        score_per_proposal_and_object = scores.max(dim=-1).values
    else:
        raise ValueError(f'Invalid {matching_strategy=}')

    # assign each proposal to the object with the highest scores
    score_per_proposal, assigned_idx_object = torch.max(score_per_proposal_and_object, dim=-1) # over n dim
    assert score_per_proposal.shape == assigned_idx_object.shape == torch.Size([N]),\
        (score_per_proposal.shape, assigned_idx_object.shape, N)

    return score_per_proposal, assigned_idx_object