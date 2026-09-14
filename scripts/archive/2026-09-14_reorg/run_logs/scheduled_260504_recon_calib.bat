@echo off
REM Scheduled task: 260504 grid reconstruction + calibration
REM Run at 2026-05-04 06:00 JST

setlocal
set SCRIPTS=C:\Users\QPI\Documents\QPI_Omni\scripts
set GRID_DIR=C:\260504\grid_2per_gluc_1
set LOGFILE=C:\260504\scheduled_recon_calib.log

echo ===== Started: %date% %time% ===== >> %LOGFILE%

REM Step 1: Update POS_SPLIT to 53 in batch_reconstruction_grid.py
python -c "p='%SCRIPTS%\\batch_reconstruction_grid.py'; t=open(p,encoding='utf-8').read(); t=t.replace('POS_SPLIT    = 52','POS_SPLIT    = 53'); open(p,'w',encoding='utf-8').write(t)"
echo POS_SPLIT updated to 53 >> %LOGFILE%

REM Step 2: Batch reconstruction (z=5 only for calibration)
echo Starting batch_reconstruction_grid.py ... >> %LOGFILE%
python %SCRIPTS%\batch_reconstruction_grid.py --grid-dir %GRID_DIR% --z-indices 5 >> %LOGFILE% 2>&1
echo batch_reconstruction_grid.py done (exit=%ERRORLEVEL%) >> %LOGFILE%

REM Step 3: Channel detection (needed for calibration)
echo Starting channel_crop.py --detect for each Pos ... >> %LOGFILE%
for /d %%D in (%GRID_DIR%\Pos*_x+0_y+0) do (
    if exist "%%D\output_phase" (
        python %SCRIPTS%\channel_crop.py --dir "%%D\output_phase" --detect >> %LOGFILE% 2>&1
    )
)
echo channel_crop.py done >> %LOGFILE%

REM Step 4: Update calibrate_grid_pos_per_pos.py parameters
python -c "
import re
p = r'%SCRIPTS%\calibrate_grid_pos_per_pos.py'
t = open(p, encoding='utf-8').read()
t = re.sub(r'BASE_DIR\s*=\s*r\"[^\"]*\"', 'BASE_DIR = r\"C:\\260504\"', t)
t = re.sub(r'TIMELAPSE_POS\s*=\s*r\"[^\"]*\"', 'TIMELAPSE_POS = r\"D:\\AquisitionData\\Kitagishi\\260504\\timelapse.pos\"', t)
t = re.sub(r'CALIB_GRID_DIR\s*=\s*r\"[^\"]*\"', 'CALIB_GRID_DIR = r\"C:\\260504\\grid_2per_gluc_1\"', t)
t = re.sub(r'REF_GRID_DIR\s*=\s*r\"[^\"]*\"', 'REF_GRID_DIR = r\"C:\\260504\\grid_2per_gluc_1\"', t)
t = re.sub(r'REF_Z_INDEX\s*=\s*\d+', 'REF_Z_INDEX   = 5', t)
t = re.sub(r'CALIB_Z_INDEX\s*=\s*\d+', 'CALIB_Z_INDEX = 5', t)
t = re.sub(r'CALIB_SUFFIX\s*=\s*\"[^\"]*\"', 'CALIB_SUFFIX   = \"x+0_y+0\"', t)
open(p, 'w', encoding='utf-8').write(t)
" >> %LOGFILE% 2>&1
echo calibrate_grid_pos_per_pos.py params updated >> %LOGFILE%

REM Step 5: Run calibration
echo Starting calibrate_grid_pos_per_pos.py ... >> %LOGFILE%
python %SCRIPTS%\calibrate_grid_pos_per_pos.py >> %LOGFILE% 2>&1
echo calibrate_grid_pos_per_pos.py done (exit=%ERRORLEVEL%) >> %LOGFILE%

echo ===== Finished: %date% %time% ===== >> %LOGFILE%
endlocal
