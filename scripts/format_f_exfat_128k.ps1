# Reformat the external backup HDD F: as exFAT with 128 KB clusters (was 2 MB: every small file used 2 MB).
# ERASES F:. Run only after scripts/restore_260928_from_f_261029.py has copied and verified 260928 back
# (restore_260928_STATUS.txt ends with "VERIFY: ... 0 missing/different").
# Needs an elevated PowerShell:  right-click PowerShell -> Run as administrator, then
#   powershell -ExecutionPolicy Bypass -File C:\Users\QPI\Documents\QPI_Omni\scripts\format_f_exfat_128k.ps1
$ErrorActionPreference = "Stop"

$status = "C:\Users\QPI\Documents\QPI_Omni\restore_260928_STATUS.txt"
if (-not (Test-Path $status) -or -not ((Get-Content $status -Tail 1) -match "VERIFY: \d+ files ok, 0 missing/different")) {
    throw "260928 restore is not verified yet ($status). F: holds the only copy of 260928 -- not formatting."
}

$v = Get-Volume -DriveLetter F
$d = Get-Partition -DriveLetter F | Get-Disk
"F: = $($v.FileSystemLabel) $($v.FileSystem) $([int]($v.Size/1GB)) GB, disk '$($d.FriendlyName)' bus $($d.BusType)"
if ($d.FriendlyName -notmatch "BUFFALO HD-PGF-A" -or $d.BusType -ne "USB" -or [int]($v.Size/1GB) -lt 4500) {
    throw "F: is not the BUFFALO HD-PGF-A USB backup disk -- not formatting."
}

Format-Volume -DriveLetter F -FileSystem exFAT -AllocationUnitSize 131072 -NewFileSystemLabel "HD-PGF-A" -Force -Confirm:$false
Get-Volume -DriveLetter F | Select-Object DriveLetter, FileSystem, AllocationUnitSize, FileSystemLabel,
    @{n='SizeGB';e={[int]($_.Size/1GB)}}, @{n='FreeGB';e={[int]($_.SizeRemaining/1GB)}} | Format-List
