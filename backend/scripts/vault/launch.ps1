# 启动脚本（双击即用）
# 密钥解析顺序：环境变量 -> 密钥箱解密 -> local.key 文件 -> 现场粘贴
# 密钥只在本次进程内存中，不落盘。
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ErrorActionPreference = 'Stop'

$key = $env:DEEPSEEK_API_KEY
if (-not $key) {
    try { $key = & "$SCRIPT_DIR\vault.ps1" get deepseek_api 2>$null } catch {}
}
if (-not $key) {
    $lk = Join-Path $SCRIPT_DIR "local.key"
    if (Test-Path $lk) { $key = (Get-Content $lk -Raw).Trim() }
}
if (-not $key) {
    $sec = Read-Host -Prompt "未找到密钥，请粘贴 DeepSeek Key（输入不可见）" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
if (-not $key -or $key.Trim().Length -eq 0) { Write-Host "无密钥，无法启动 LLM 模式。"; exit 1 }
$env:DEEPSEEK_API_KEY = $key.Trim()

$backendDir = Join-Path $SCRIPT_DIR ".." ".."
$proxy = Join-Path $SCRIPT_DIR "proxy.py"
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"Set-Location '$backendDir'; uvicorn app.main:app --host 127.0.0.1 --port 8200`"" -WindowStyle Minimized
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"python '$proxy'`"" -WindowStyle Minimized
Write-Host "HealthPick 已启动：后端 8200，前端代理 8201。"
Write-Host "打开 http://127.0.0.1:8201  （密钥仅在本次进程内存中，不落盘）"
