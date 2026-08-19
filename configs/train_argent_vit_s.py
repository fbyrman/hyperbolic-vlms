from .train_argent_vit_b import dataset, model, optim, train  # noqa: F401

model.visual.arch = "vit_small_mocov3_patch16_224"
