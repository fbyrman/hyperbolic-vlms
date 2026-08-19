# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# "Euclidean inference" ablation: zero-shot classification on a hyperbolic-trained
# checkpoint (MERU/HyCoCLIP/ARGENT/PHyCLIP), scored in the tangent/ambient space
# instead of on the hyperboloid -- i.e. encode with `project=False`, scale by the
# learned alpha(s), skip `exp_map0`, and score by negative Euclidean distance.
# See `ZeroShotClassificationEvaluator(eval_mode="ambient")`. Meaningless for
# CLIPBaseline/Euclidean* checkpoints, which have no exponential map to remove.
#
# The suite and prompts are the default config's, so the two runs stay
# comparable dataset for dataset.

from gvlm.config import LazyCall as L
from gvlm.evaluation.classification import ZeroShotClassificationEvaluator

from .eval_zero_shot_classification import evaluator as _default

evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts=_default.datasets_and_prompts,
    data_dir=_default.data_dir,
    image_size=_default.image_size,
    eval_mode="ambient",
)
