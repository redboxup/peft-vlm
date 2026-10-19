---
name: add-method
description: Add a new prompt-learning method (config, model, trainer, optional loss) to this repo. Use when the user wants to implement or register a new trainer/method such as a new CoOp variant.
---

A method is wired together by one name string (`<name>`, lowercase, e.g. `foo`) in four places. Do all of them, then verify.

1. **Config** - in `allconfigs/train_configs.py` add `FooConf(TrainConf)` (copy the closest existing conf, e.g. `CoOpConf`). In `configs.py` import it and add `cs.store(name='<name>', group='trainer', node=FooConf)`. The `name` field default in the dataclass must equal `<name>`.
2. **Model** - `models/Foo.py`: subclass `BaseCLIP`, decorate with `@BaseCLIP.register_model("<name>")`, implement `forward` (and `forward_feats` if the trainer needs local features). Add `from .Foo import FooCLIP` to `models/__init__.py`. If the method needs a modified CLIP backbone, add `clip/model_foo.py` and branch on `model_i` in `clip/clip.py:load`.
3. **Trainer** - `trainers/Foo.py`: subclass `BaseTrainer` (or the closest existing trainer, as `LoCoOpTrainer` extends `CoOpTrainer`), decorate with `@BaseTrainer.register_trainer("<name>")`, override `get_losses`, `model_forward`, `compute_loss`. Add the import to `trainers/__init__.py`.
4. **Loss (optional)** - `losses/Foo.py`, returned from the trainer's `get_losses()`.

Reuse is fine: several names can share one model class by stacking `register_model` decorators (see `models/CoOp.py`, which serves `coop` and `locoop`).

Verify:
- `python3 -m py_compile` on every new/edited file.
- `python3 -c "import models, trainers"`. NOTE: this repo's `__init__.py` files currently import modules that do not exist (`manual_prompt`, `MoCoOp`, `DualCoOpPP`, `Ensemble_prompt1/2/3`, `vision_classifier`); if the import fails only on those, tell the user rather than silently deleting them.
- `python3 configs.py +dataset=chexpert +trainer=<name>` prints the composed config without needing a GPU.
- A real run needs CUDA and `DATAPATH`; see the `run-experiment` skill.

Also add the method to the README "Supported Methods" table and `docs/architecture.md` methods table.
