# HealthPick 一键启动器：进入交互端（密钥直接编辑 deploy/.env 填写，本脚本只读取不询问）
# 功能：① 读取 deploy/.env 密钥（有→真 AI；无→本地规则降级）→ ② 自动启动（Docker 优先，Python 兜底）→ ③ 打开浏览器
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
Write-Host "   一键启动（双击即用 / 换电脑也能用）" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# ── 1) 读取密钥（只读，不询问）────────────────────────────
$existing = ""
if (Test-Path $EnvFile) {
    $envContent = Get-Content $EnvFile -Raw
    if ($envContent -match 'DEEPSEEK_API_KEY=(\S+)') { $existing = $matches[1] }
}
Write-Host ""
if ($existing) {
    $masked = $existing.Substring(0, [Math]::Min(8, $existing.Length))
    Write-OK ("已读取密钥（" + $masked + "...），启动真 AI 模式")
} else {
    Write-Warn "未配置密钥：将用本地规则模式启动（功能完整）。如需真 AI，请打开 deploy\.env 文件，填入 DEEPSEEK_API_KEY=你的密钥。"
}

# ── 2) 启动服务（Docker 优先，Python 兜底）─────────────────
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
    if ($existing) { $env:DEEPSEEK_API_KEY = $existing }
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

# ── 3) 自检 + 打开浏览器 ───────────────────────────────────
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
Write-Host "如需更换密钥，请编辑 deploy\.env 文件后重新双击本脚本。"
Write-Host ""
Read-Host "按回车退出"
