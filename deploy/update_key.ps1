# HealthPick 更新密钥脚本：粘贴新密钥 → 覆盖写入 deploy/.env（下次启动自动生效）
# 独立于启动器：只做密钥更新，不启动服务。
$ErrorActionPreference = 'Stop'

$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EnvFile = Join-Path $DeployDir ".env"

function Write-OK($m) { Write-Host ("OK  " + $m) -ForegroundColor Green }
function Write-Warn($m) { Write-Host ("!!  " + $m) -ForegroundColor Yellow }

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   HealthPick 健康优选 - 更新密钥" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 读取当前密钥（只显示前缀，不显示明文）
$existing = ""
if (Test-Path $EnvFile) {
    $envContent = Get-Content $EnvFile -Raw
    if ($envContent -match 'DEEPSEEK_API_KEY=(\S+)') { $existing = $matches[1] }
}
Write-Host ""
if ($existing) {
    $masked = $existing.Substring(0, [Math]::Min(8, $existing.Length))
    Write-OK ("当前密钥：" + $masked + "...")
    Write-Warn "粘贴新密钥回车 = 覆盖更新；直接回车 = 取消（密钥不变）"
} else {
    Write-Warn "尚未配置密钥。"
}
Write-Host ""
$key = Read-Host "粘贴新的 DeepSeek API Key（回车取消）"
$key = ($key -as [string]).Trim()
if ($key) {
    # 覆盖写入 deploy/.env（UTF-8 无 BOM，docker compose 兼容）
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($EnvFile, "DEEPSEEK_API_KEY=$key`r`n", $utf8NoBom)
    Write-OK "密钥已更新（覆盖写入 deploy\.env）"
    Write-Host "下次双击「一键启动」即生效。"
} else {
    Write-Warn "未输入，密钥保持不变。"
}
Write-Host ""
Read-Host "按回车退出"
