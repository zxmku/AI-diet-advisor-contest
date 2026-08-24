# HealthPick 一键启动器（离线自包含版）
# ① 读取 deploy/.env 密钥（有→真 AI；无→本地规则降级）
# ② 用本机 Python 起服务；依赖已随仓库离线打包（deploy/wheels），无需联网
# ③ 就绪后打开浏览器
#
# 设计目标（针对"干净 Windows 双击没反应 / 联网下载崩溃"的根因）：
#   - 零外部依赖：依赖装包来自 deploy/wheels（仓库自带），启动全程不访问 PyPI / 外网
#   - 错误全程可见，绝不静默吞掉（pip / uvicorn 失败都会打印并停留）
#   - Python 缺失时给明确安装指引并停留（不擅自联网装，避免墙内/离线失败）
#   - 启动后轮询 /health，失败读日志报错，绝不盲开一个死端口
#   - 路径全部由脚本所在目录推导，零硬编码，换电脑可用。

$DeployDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir    = Split-Path -Parent $DeployDir
$EnvFile    = Join-Path $DeployDir ".env"
$WheelsDir  = Join-Path $DeployDir "wheels"
$BackendDir = Join-Path $RootDir "backend"
$PORT       = 8137
$LogDir     = $DeployDir

function Write-Step($m){ Write-Host ("[*] " + $m) -ForegroundColor Cyan }
function Write-OK($m){ Write-Host ("[+] " + $m) -ForegroundColor Green }
function Write-Warn($m){ Write-Host ("[!] " + $m) -ForegroundColor Yellow }
function Write-Err($m){ Write-Host ("[X] " + $m) -ForegroundColor Red }

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   HealthPick · AI 智能膳食顾问 · 一键启动（离线版）" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  ☁️ 推荐优先使用 README 顶部的「线上体验」链接（免安装、体验更稳）" -ForegroundColor Yellow
Write-Host "     本脚本为本地兜底，需本机已装 Python 3.10-3.12" -ForegroundColor Yellow

# ── 1) 读取密钥（只读，不询问）────────────────────────────
$apiKey = ""
if (Test-Path $EnvFile) {
    $c = Get-Content $EnvFile -Raw
    if ($c -match 'DEEPSEEK_API_KEY=(\S+)') { $apiKey = $matches[1] }
}
if ($apiKey) {
    # 2026-08-24 密钥纪律加固：绝不打印密钥任何片段（演示/录屏即泄露），只报状态
    Write-OK "已检测到 DeepSeek 密钥（已配置），将启动真 AI 模式"
} else {
    Write-Warn "未检测到密钥：将以本地规则模式运行（核心功能完整，仅无 AI 润色）。"
    Write-Warn "如需真 AI，用记事本打开 deploy\.env，填 DEEPSEEK_API_KEY=你的密钥 后重双击。"
}

# ── 2) 查找本机 Python（仅支持 3.10/3.11/3.12，因离线依赖包只内置这三版；不联网安装）──
Write-Step "查找本机 Python 运行环境（需 3.10 / 3.11 / 3.12）..."
function Get-Python {
    foreach ($cmd in @("python", "py", "python3")) {
        $p = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($p) {
            $ver = & $p.Source -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
            if ($ver -match '^3\.1[0-2]$') { return $p.Source }
        }
    }
    return $null
}
$py = Get-Python
if (-not $py) {
    Write-Err "本机未找到 Python 3.10 / 3.11 / 3.12，无法启动。"
    Write-Err "（离线依赖包 deploy\wheels 仅内置这三版的安装包，其他版本如 3.13 暂不支持）"
    Write-Err "请先安装 Python 3.11（一次即可，属于运行环境）："
    Write-Err "  https://www.python.org/downloads/release/python-3119/  （安装时务必勾选 'Add to PATH'）"
    Write-Err "装好后重新双击本脚本即可。本程序不联网自动安装，避免墙内/离线失败。"
    Read-Host "按回车退出"
    exit 1
}
Write-OK ("使用 Python：" + $py)

# ── 3) 准备依赖（离线：从仓库自带的 deploy/wheels 安装，绝不访问外网）──
Write-Step "准备运行环境（离线安装自带依赖，无需联网）..."
$venv      = Join-Path $BackendDir ".venv"
$pipLog    = Join-Path $LogDir "pip_install.log"
if (-not (Test-Path $venv)) {
    Write-Step "创建虚拟环境 $venv ..."
    & $py -m venv $venv 2>&1 | Out-File -Append -FilePath $pipLog -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Err "创建虚拟环境失败，详见 $pipLog"
        Write-Err (Get-Content $pipLog -Tail 15 -ErrorAction SilentlyContinue | Out-String)
        Read-Host "按回车退出"
        exit 1
    }
}
$pyExe = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $pyExe)) { $pyExe = $py }   # 回退：直接用系统 python

if (Test-Path $WheelsDir) {
    Write-Step ("离线安装依赖（来源：$WheelsDir，日志：$pipLog）...")
    & $pyExe -m pip install --no-index --find-links $WheelsDir -r (Join-Path $BackendDir "requirements.txt") *> $pipLog
} else {
    Write-Warn "未找到 deploy/wheels，退回联网安装（需网络）："
    & $pyExe -m pip install -r (Join-Path $BackendDir "requirements.txt") *> $pipLog
}
if ($LASTEXITCODE -ne 0) {
    Write-Err "依赖安装失败，最后 15 行日志："
    Write-Err (Get-Content $pipLog -Tail 15 -ErrorAction SilentlyContinue | Out-String)
    Write-Err "若为离线环境，请确认 deploy/wheels 目录已随仓库完整存在。"
    Read-Host "按回车退出"
    exit 1
}
Write-OK "依赖就绪（离线）"

# ── 4) 启动服务（后台运行，日志落盘，绝不静默）──────────
if ($apiKey) { $env:DEEPSEEK_API_KEY = $apiKey }
Write-Step ("启动服务（端口 $PORT，日志：server.out.log / server.err.log）...")
$outLog = Join-Path $LogDir "server.out.log"
$errLog = Join-Path $LogDir "server.err.log"
$proc = Start-Process -FilePath $pyExe `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", $PORT `
    -WorkingDirectory $BackendDir `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
    -PassThru -WindowStyle Hidden

# ── 5) 轮询 /health，就绪才开浏览器 ────────────────────
Write-Step "等待服务就绪（最多 60 秒）..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$PORT/health" -TimeoutSec 3
        $ready = $true; break
    } catch { }
}
if ($ready) {
    Write-OK "服务已就绪！"
    Start-Process "http://localhost:$PORT"
    Write-Host ("已打开交互页：http://localhost:" + $PORT) -ForegroundColor Green
} else {
    Write-Err "服务在 60 秒内未就绪。最后错误日志："
    Write-Err (Get-Content $errLog -Tail 20 -ErrorAction SilentlyContinue | Out-String)
    Write-Err "请按上方错误排查，或改用命令行启动（见 README「快速开始」）。"
}
Write-Host ""
Write-Host "提示：关闭本窗口不会停止服务；如需停止，结束 uvicorn 进程即可。" -ForegroundColor DarkGray
Read-Host "按回车退出（服务在后台继续运行）"
