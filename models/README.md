# models/ - Omnipose checkpoints used for production masks

Each checkpoint that produced a published master lives here under its short name
(`omni_model_d20_<train timestamp>`), with its sha256, training date, training set and the
exact `eval` parameters in `MODELS.json`. `datasets/<id>.yaml` refers to a checkpoint by this
path, `run_dataset_pipeline.py` copies it into every master it publishes, and
`environment/check_env.py --model models/<file>` verifies the hash on a new machine.

Adding a checkpoint after training (`scripts/08_train.py` writes into
`C:\Users\QPI\Desktop\train\omni_model_d20\models\` with the long cellpose name):

1. pick the epoch with `checkpoint_eval.py` / `checkpoint_overlay_runner.py`
2. copy it here as `omni_model_d20_<timestamp>` (the long name exceeds Windows MAX_PATH inside masters)
3. add an entry to `MODELS.json` (sha256 from `sha256sum`, bytes, trained, train_set, eval)
4. point the dataset yaml at it

A checkpoint is 26.6 MB, so the repository grows by that much per model; if several are added,
move to git LFS. Old training runs stay outside the repository.
