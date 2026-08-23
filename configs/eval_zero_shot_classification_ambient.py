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
from gvlm.evaluation.classification import ZeroShotClassificationEvaluator

from .eval_zero_shot_classification import evaluator as _default

evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts=_default.datasets_and_prompts,
    data_dir=_default.data_dir,
    image_size=_default.image_size,
    eval_mode="ambient",
)
