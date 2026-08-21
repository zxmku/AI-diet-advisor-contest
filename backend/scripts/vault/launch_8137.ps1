# 评委口 8137 启动器（双击即用；密钥经 run_hp.py 注入进程内存，明文不常驻）
$ErrorActionPreference = 'Stop'
$VaultDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$backendDir = Join-Path $VaultDir "..\.."
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "python" }
$runner = Join-Path $VaultDir "run_hp.py"
$keyfile = Join-Path $env:TEMP "hp_key.tmp"
$LOG = Join-Path $env:TEMP "launch8137.log"
function Log($m) { $m | Out-File -Append $LOG }

"" | Set-Content $LOG
Log ("[{0}] 启动 8137" -f (Get-Date).ToString('HH:mm:ss'))

# 1) 解析密钥：环境变量 -> 密钥箱 -> local.key（不交互、不回显）
$key = $env:DEEPSEEK_API_KEY
if (-not $key) { try { $key = (& (Join-Path $VaultDir "vault.ps1") get deepseek_api 2>$null).Trim() } catch {} }
if (-not $key) { $lk = Join-Path $VaultDir "local.key"; if (Test-Path $lk) { $key = (Get-Content $lk -Raw).Trim() } }
if (-not $key) { Log "KEY_MISSING"; Write-Host "无密钥，无法启动 LLM 模式。"; exit 1 }
$key | Set-Content -Path $keyfile -Encoding ASCII
Log ("KEY_OK len=" + $key.Length)

# 2) 清理占用 8137 的陈旧实例（netstat 解析端口，比 Get-NetTCPConnection 可靠）
try {
    $line = netstat -ano | Select-String ":8137\s+.*LISTENING" | Select-Object -First 1
    if ($line) {
        $pid8137 = ($line.ToString().Trim() -split '\s+')[-1]
        Log ("KILL_STALE pid=" + $pid8137)
        Stop-Process -Id $pid8137 -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    } else { Log "NO_STALE" }
} catch { Log ("CLEAN_ERR " + $_.Exception.Message) }

# 3) 以 python 直启当前构建于 8137（前端由 main.py 静态挂载，同端口提供 API+UI）
Start-Process -FilePath $py -ArgumentList $runner, $keyfile -WorkingDirectory $backendDir -WindowStyle Minimized
Log "START_SENT"
Start-Sleep -Seconds 5

# 4) 自检
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8137/health"
    Log ("HEALTH llm_key_set=" + $r.llm_key_set + " model=" + $r.llm_model)
} catch { Log ("HEALTH_ERR " + $_.Exception.Message) }
Write-Host "HealthPick 评委口已启动：http://127.0.0.1:8137"
Log "DONE"
