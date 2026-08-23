# Modified from github.com/tksmatsubara/PHyCLIP

import torch
from torch.nn import functional as F

from gvlm import euclidean as E
from gvlm import lorentz as L
from gvlm.models import (
    MERU,
    CLIPBaseline,
    EuclideanMERU,
    EuclideanPhyCLIP,
    HyCoCLIP,
    PHyCLIP,
)


def process_text_features(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Process text features based on model type.

    Args:
        model: The model instance
        text_features: Raw text features from the model

    Returns:
        Processed text features ready for similarity computation
    """
    if isinstance(model, PHyCLIP):
        # map class features to each concept's hyperboloid
        class_feats = text_features.mean(dim=0).view(
            -1, model.num_subspaces, model.subspace_dim
        )

        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)

        curvs = torch.stack([curv.exp() for curv in model.curvs])
        class_feats = L.exp_map0_batch(class_feats, curvs).view(model.num_subspaces, -1)
        return class_feats
    elif isinstance(model, EuclideanPhyCLIP):
        # Before EuclideanMERU: EuclideanPhyCLIP subclasses it.
        class_feats = text_features.mean(dim=0).view(
            -1, model.num_subspaces, model.subspace_dim
        )

        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)

        class_feats = E.exp_map0_batch(class_feats, model.eps)
        return class_feats.view(model.num_subspaces, -1)
    elif isinstance(model, (HyCoCLIP, MERU)):
        # Ensemble in the tangent space, then project to Hyperboloid.
        class_feats = text_features.mean(dim=0)
        class_feats = class_feats * model.textual_alpha.exp()
        class_feats = L.exp_map0(class_feats, model.curv.exp())
        return class_feats
    elif isinstance(model, EuclideanMERU):
        class_feats = text_features.mean(dim=0)
        class_feats = class_feats * model.textual_alpha.exp()
        return E.exp_map0(class_feats, model.eps)
    else:
        # Ensemble prompt features: normalize -> average -> normalize.
        class_feats = F.normalize(text_features, dim=-1)
        class_feats = class_feats.mean(dim=0)
        class_feats = F.normalize(class_feats, dim=-1)
        return class_feats


def process_text_features_ambient(
    model: PHyCLIP | HyCoCLIP | MERU,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Ambient-space counterpart of `process_text_features`, used to build a
    zero-shot classifier for `eval_mode="ambient"`: ensemble prompts in the
    tangent space and scale by the learned textual alpha(s), same as
    `to_ambient_features`, but skip the exponential map. Scaling is linear so
    averaging prompts first (as here) or scaling each prompt before
    averaging (as `to_ambient_features` does per-instance) are equivalent.

    Args:
        model: A MERU, HyCoCLIP, ARGENT, or PHyCLIP instance.
        text_features: Raw per-prompt text features, `project=False`.
    """
    if isinstance(model, PHyCLIP):
        class_feats = text_features.mean(dim=0).view(
            -1, model.num_subspaces, model.subspace_dim
        )
        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)
        return class_feats.view(model.num_subspaces, -1)
    else:
        class_feats = text_features.mean(dim=0)
        return class_feats * model.textual_alpha.exp()


def to_ambient_features(
    model: PHyCLIP | HyCoCLIP | MERU,
    feats: torch.Tensor,
    modality: str,
) -> torch.Tensor:
    """
    Scale raw (`project=False`) encoder features by the model's learned
    visual/textual alpha(s) -- the same scaling applied before `exp_map0` --
    but skip the exponential map itself. This is the tangent-space (ambient)
    representation used for the "Euclidean inference" ablation: retrieval on
    a hyperbolic-trained checkpoint scored with Euclidean distance instead of
    the Lorentzian geometry it was trained with.

    Args:
        model: A MERU, HyCoCLIP, ARGENT, or PHyCLIP instance (never a
            `Euclidean*` model -- those are already ambient by construction).
        feats: Output of `model.encode_{image,text}(..., project=False)`.
        modality: Either `"image"` or `"text"`.
    """
    if isinstance(model, PHyCLIP):
        alphas = (
            model.visual_concept_alphas
            if modality == "image"
            else model.textual_concept_alphas
        )
        feats = feats.view(-1, model.num_subspaces, model.subspace_dim)
        feats = feats * torch.stack([a.exp() for a in alphas]).view(1, -1, 1)
        return feats.view(-1, model.embed_dim)

    alpha = model.visual_alpha if modality == "image" else model.textual_alpha
    return feats * alpha.exp()


def compute_ambient_similarity_scores(
    model: PHyCLIP | HyCoCLIP | MERU,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Score two tangent-space (ambient) feature batches -- see
    `to_ambient_features` -- with negative Euclidean distance.

    For `PHyCLIP` (and `ARGENT`'s multi-subspace cousins), distances are
    computed per concept subspace and combined with the model's own
    `product_metric`, mirroring `EuclideanPhyCLIP._combine` exactly so the
    only thing that changed relative to training is the removal of the
    per-subspace exponential map.
    """
    if isinstance(model, PHyCLIP):
        image_features = image_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        distances = E.pairwise_dist_batch(
            image_features, text_features, eps=model.lorentz_eps
        )
        if model.product_metric == "l1":
            combined = distances.mean(dim=0)
        else:
            combined = torch.sqrt((distances**2).mean(dim=0))
        return -combined

    return -E.pairwise_dist(image_features, text_features)


def compute_similarity_scores(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute similarity scores between image and text features based on model type.

    Args:
        model: The model instance
        image_features: Image features
        text_features: Processed text features (classifier)

    Returns:
        Similarity scores
    """

    if isinstance(model, PHyCLIP):
        curvs = torch.stack([curv.exp() for curv in model.curvs])
        image_features = image_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        scores = L.pairwise_inner_batch(image_features, text_features, curvs).sum(dim=0)
        return scores
    elif isinstance(model, EuclideanPhyCLIP):
        # Negative distance, combined across subspaces the way training does it.
        image_features = image_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        distances = E.pairwise_dist_batch(image_features, text_features, eps=model.eps)
        return -model._combine(distances)
    elif isinstance(model, (HyCoCLIP, MERU)):
        scores = L.pairwise_inner(image_features, text_features, model.curv.exp())
        return scores
    elif isinstance(model, EuclideanMERU):
        return -E.pairwise_dist(image_features, text_features, eps=model.eps)
    else:
        scores = image_features @ text_features.T
        return scores
