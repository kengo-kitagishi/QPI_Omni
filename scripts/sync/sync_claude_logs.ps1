# sync_claude_logs.ps1
# Claude会話ログ（.jsonl）をGoogle Driveに毎日同期する

$src = "C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni"
$dst = "G:\共有ドライブ\wakamotolab_meeting\kitagishi\claude_logs\QPI_Omni"
$logFile = "C:\Users\QPI\.claude\sync_log.txt"

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# 同期先ディレクトリがなければ作成
if (-not (Test-Path $dst)) {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
}

# robocopy で .jsonl のみコピー（memory/ フォルダは除外）
robocopy $src $dst "*.jsonl" /XD memory /NP /NDL /NC /NJS /NJH | Out-Null
$exitCode = $LASTEXITCODE

# robocopy の終了コード: 0=変更なし, 1=コピー成功, 2以上=一部エラー or 警告
if ($exitCode -le 1) {
    $status = "OK"
} else {
    $status = "WARN(exit=$exitCode)"
}

# ログ追記
"$timestamp [$status] $src -> $dst" | Out-File -FilePath $logFile -Append -Encoding UTF8

Write-Host "$timestamp [$status] Sync done."
