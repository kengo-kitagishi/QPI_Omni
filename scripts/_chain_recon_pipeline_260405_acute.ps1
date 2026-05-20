$ErrorActionPreference = 'Continue'
$ROOT = 'F:\260405_acute_z18_200h'
$SCRIPTS = 'C:\Users\QPI\Documents\QPI_omni\scripts'
$LOG = 'C:\Users\QPI\Documents\QPI_omni\scripts\_chain_recon_pipeline_260405_acute.log'

function Log($msg) {
    $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $line = "[$stamp] $msg"
    Write-Output $line
    Add-Content -Path $LOG -Value $line
}

Set-Content -Path $LOG -Value ("=== chain_recon_pipeline_260405_acute START " + (Get-Date) + " ===")

Log 'Step A2: grid_0per recon (z=18)'
& python "$SCRIPTS\batch_reconstruction_grid.py" --grid-dir "$ROOT\grid_0pergluc_60ms_1" --z-indices 18 2>&1 | Tee-Object -FilePath $LOG -Append
if ($LASTEXITCODE -ne 0) { Log "ABORT: grid_0per recon failed (exit $LASTEXITCODE)"; exit 1 }
Log 'Step A2 done.'

Log 'Step B+C: full batch_pipeline_all_pos.py --skip-grid-0per (timelapse recon + ECC + grid_subtract + correct_0pergluc, Pos1..Pos64)'
& python "$SCRIPTS\batch_pipeline_all_pos.py" --skip-grid-0per 2>&1 | Tee-Object -FilePath $LOG -Append
if ($LASTEXITCODE -ne 0) { Log "ABORT: batch_pipeline_all_pos failed (exit $LASTEXITCODE)"; exit 1 }
Log 'Step B+C done.'

Log 'Step D: per-Pos analysis (seg + lineage tracker + per_channel_figures + batch_figures)'
$cal = "$ROOT\grid_2pergluc_60ms_1\ri_calibration_results.json"
$schedule = '0:wo_2,575:wo_0,1440:wo_2'
$pos_dirs = Get-ChildItem "$ROOT\ph_260405" -Directory -Filter 'Pos*' |
    Where-Object { (Test-Path "$($_.FullName)\output_phase\channels\crop_sub_rawraw") } |
    Sort-Object {[int]([regex]::Match($_.Name,'Pos(\d+)').Groups[1].Value)}
Log "  found $($pos_dirs.Count) Pos dirs with crop_sub_rawraw output"
foreach ($pd in $pos_dirs) {
    $root_for_chs = "$($pd.FullName)\output_phase\channels\crop_sub_rawraw"
    Log "  --- analyse $($pd.Name) --- root=$root_for_chs"
    & python "$SCRIPTS\batch_all_channels.py" `
        --root $root_for_chs `
        --ri-calibration $cal `
        --media-schedule $schedule 2>&1 | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) { Log "  WARN: batch_all_channels failed for $($pd.Name) (exit $LASTEXITCODE), continuing" }
}
Log 'Step D done.'

Log '=== chain_recon_pipeline_260405_acute END ==='
