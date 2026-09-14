# datasets/ - one yaml per experiment

Everything that differs between experiments (where the phase crops and masks live, the medium
switch frames, the RI calibration, the drift bad frames, the Omnipose checkpoint, the tracker
constants) is written once in `datasets/<YYMMDD>.yaml`. `scripts/run_dataset_pipeline.py` reads
it and runs segmentation -> tracking -> division QC -> consolidation -> master publication, and
copies the yaml into the master as provenance.

```powershell
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --plan        # what would run
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml               # seg,track,qc,consolidate
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --stages consolidate,publish --tag v20261001_first
```

`TEMPLATE.yaml` lists every field with its meaning; `260517.yaml` is the filled example.
The acquisition side (grid calibration, drift session, `compute_drift_online.py`,
`correct_0pergluc.py`) still runs from `docs/PROTOCOL_TIMELAPSE.md` sections 2-8 and produces the
`raw_root` this yaml points at.
