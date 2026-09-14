@echo off
echo [%DATE% %TIME%] START reconstruction
C:\Users\QPI\AppData\Local\Programs\Python\Python311\python.exe C:\Users\QPI\Documents\QPI_Omni\scripts\batch_reconstruction_grid.py
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: reconstruction failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [%DATE% %TIME%] START grid calibration
C:\Users\QPI\AppData\Local\Programs\Python\Python311\python.exe C:\Users\QPI\Documents\QPI_Omni\scripts\calibrate_grid_pos_per_pos.py
echo [%DATE% %TIME%] DONE
