# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# The default config's suite, scored in the tangent space instead of on the
# hyperboloid: `project=False`, skip `exp_map0`, rank by negative Euclidean
# distance. Inert on `CLIPBaseline` and the `Euclidean*` controls, which have no
# exponential map to remove.

from gvlm.config import LazyCall as L
from gvlm.evaluation.retrieval import ZeroShotRetrievalEvaluator

evaluator = L(ZeroShotRetrievalEvaluator)(
    datasets=["coco", "flickr30k"],
    data_dir="datasets/eval",
    image_size=224,
    eval_mode="ambient",
)
