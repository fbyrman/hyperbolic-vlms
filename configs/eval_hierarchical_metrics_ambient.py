# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# "Euclidean inference" ablation: hierarchical metrics on a hyperbolic-trained
# checkpoint (MERU/HyCoCLIP/ARGENT/PHyCLIP), scored in the tangent/ambient
# space instead of on the hyperboloid -- i.e. encode with `project=False`,
# scale by the learned alpha(s), skip `exp_map0`, and rank by negative
# Euclidean distance. See `HierarchicalMetricsEvaluator(eval_mode="ambient")`.
# Meaningless for CLIPBaseline/Euclidean* checkpoints, which have no
# exponential map to remove.

from gvlm.config import LazyCall as L
from gvlm.evaluation.hierarchical_metrics import HierarchicalMetricsEvaluator

evaluator = L(HierarchicalMetricsEvaluator)(
    datasets_and_prompts={
        "imagenet": [
            "i took a picture : itap of a {}.",
            "pics : a bad photo of the {}.",
            "pics : a origami {}.",
            "pics : a photo of the large {}.",
            "pics : a {} in a video game.",
            "pics : art of the {}.",
            "pics : a photo of the small {}.",
        ]
    },
    data_dir="datasets/eval",
    image_size=224,
    eval_mode="ambient",
)
