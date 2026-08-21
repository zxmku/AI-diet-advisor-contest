# 更新密钥脚本（用户专用）
# 用法：双击本脚本 -> 粘贴 DeepSeek Key（输入不可见）-> 自动用 vault.pub 加密落盘为 deepseek_api.vsec
# 明文只在这一瞬间存在于内存，绝不写入任何文件 / 聊天 / 日志。
# 重跑即更新（覆盖重加密）。
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$sec = Read-Host -Prompt "请粘贴你的 DeepSeek API Key（输入不可见）" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
$key = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
if (-not $key -or $key.Trim().Length -eq 0) { Write-Host "未输入，已取消。"; exit 1 }
$key.Trim() | & "$SCRIPT_DIR\vault.ps1" add deepseek_api -Force
Write-Host "OK - Secret stored successfully（密文已保存到 deepseek_api.vsec，明文不落盘）"
