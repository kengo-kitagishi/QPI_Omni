# register_task.ps1 - タスクスケジューラに毎日22:00の同期タスクを登録する

$python = "C:\Users\QPI\AppData\Local\Programs\Python\Python311\python.exe"
$script = "C:\Users\QPI\Documents\QPI_Omni\scripts\sync\sync_claude_logs.py"

# 既存タスクを削除
$existing = Get-ScheduledTask -TaskName "SyncClaudeLogs" -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Unregister-ScheduledTask -Confirm:$false
    Write-Host "Old task removed."
}

$action = New-ScheduledTaskAction -Execute $python -Argument $script

$trigger = New-ScheduledTaskTrigger -Daily -At "22:00"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName "SyncClaudeLogs" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Claude JSONL logs -> Google Drive daily sync (Python)" `
    -Force

Write-Host "Task registered: SyncClaudeLogs (daily 22:00)"
