"""部署引导：从临时文件读取密钥（启动即删，明文不常驻），注入环境后拉起 uvicorn。

用法：python run_hp.py <key_tmp_file>
密钥仅在进程内存中，文件读取后立即删除，避免明文落盘常驻。
"""
import os
import sys
import uvicorn

sys.path.insert(0, os.getcwd())

KEY_FILE = sys.argv[1] if len(sys.argv) > 1 else ""
if KEY_FILE and os.path.exists(KEY_FILE):
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        os.environ["DEEPSEEK_API_KEY"] = f.read().strip()
    try:
        os.remove(KEY_FILE)
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8137)
