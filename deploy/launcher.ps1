# HealthPick 交互端一键启动器（评委/用户双击即用）
# 功能：① 设置/更新 DeepSeek API Key（覆盖式）→ ② 自动启动（Docker 优先，Python 兜底）→ ③ 打开浏览器
# 换电脑可用：所有路径由脚本所在目录推导，零硬编码物理路径。
$ErrorActionPreference = 'Stop'

$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $DeployDir
$EnvFile = Join-Path $DeployDir ".env"
$BackendDir = Join-Path $RootDir "backend"
$PORT = 8137

function Write-Step($m) { Write-Host $m -ForegroundColor Cyan }
function Write-OK($m) { Write-Host ("OK  " + $m) -ForegroundColor Green }
function Write-Warn($m) { Write-Host ("!!  " + $m) -ForegroundColor Yellow }

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   HealthPick 健康优选 · AI 智能膳食顾问" -ForegroundColor Cyan
Write-Host "   一键启动器（双击即用 / 换电脑也能用）" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# ── 1) 密钥设置（覆盖式更新）────────────────────────────────────
$existing = ""
if (Test-Path $EnvFile) {
    $envContent = Get-Content $EnvFile -Raw
    if ($envContent -match 'DEEPSEEK_API_KEY=(\S+)') { $existing = $matches[1] }
}
Write-Host ""
if ($existing) {
    $masked = $existing.Substring(0, [Math]::Min(8, $existing.Length))
    Write-OK ("当前密钥已配置：" + $masked + "...")
    Write-Warn "直接回车 = 保留当前密钥；粘贴新密钥回车 = 覆盖更新"
} else {
    Write-Warn "尚未配置 API Key，将尝试以本地规则模式启动（功能完整）。"
}
Write-Host ""
$key = Read-Host "粘贴 DeepSeek API Key（无 Key 直接回车）"
$key = ($key -as [string]).Trim()
if ($key) {
    # 覆盖写入 deploy/.env（UTF-8 无 BOM，docker compose 兼容）
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($EnvFile, "DEEPSEEK_API_KEY=$key`r`n", $utf8NoBom)
    Write-OK "密钥已更新（覆盖写入 deploy\.env）"
} elseif (-not $existing) {
    Write-Warn "未输入密钥，将以本地规则模式启动。"
} else {
    Write-OK "保留当前密钥。"
}

# ── 2) 启动服务（Docker 优先，Python 兜底）─────────────────────
$docker = Get-Command docker -ErrorAction SilentlyContinue
$py = Get-Command python -ErrorAction SilentlyContinue
$started = $false

Write-Host ""
if ($docker) {
    Write-Step "检测到 Docker -> 使用容器方式启动（推荐）..."
    Push-Location $DeployDir
    try {
        docker compose up -d
        $started = $true
    } catch {
        Write-Warn ("Docker 启动失败：" + $_.Exception.Message)
        Write-Warn "尝试 Python 本地方式..."
    }
    Pop-Location
}
if (-not $started -and $py) {
    Write-Step "检测到 Python -> 使用本地方式启动..."
    if (Test-Path $EnvFile) {
        $envContent = Get-Content $EnvFile -Raw
        if ($envContent -match 'DEEPSEEK_API_KEY=(\S+)') { $env:DEEPSEEK_API_KEY = $matches[1] }
    }
    Push-Location $BackendDir
    & $py.Source -m pip install -q -r requirements.txt 2>$null
    Start-Process -FilePath $py.Source -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", $PORT -WorkingDirectory $BackendDir -WindowStyle Minimized
    Pop-Location
    $started = $true
}
if (-not $started) {
    Write-Warn "未检测到 Docker 或 Python。请先安装 Docker Desktop（推荐），或安装 Python 3.10+ 后重试。"
    Read-Host "按回车退出"
    exit 1
}

# ── 3) 自检 + 打开浏览器 ───────────────────────────────────────
Write-Step "等待服务就绪..."
$ready = $false
for ($i = 0; $i -lt 10; $i++) {
    Start-Sleep -Seconds 2
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/health" -TimeoutSec 3
        $ready = $true
        break
    } catch { }
}
if ($ready) { Write-OK "服务已就绪！" }
else { Write-Warn "服务启动中（浏览器将自动打开）" }
Start-Process "http://localhost:$PORT"
Write-Host ""
Write-Host ("已打开交互页面：http://localhost:" + $PORT) -ForegroundColor Green
Write-Host "如需更换密钥：再次双击本启动器，粘贴新密钥即可覆盖更新。"
Write-Host ""
Read-Host "按回车退出"
