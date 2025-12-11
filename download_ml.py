import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. 强制使用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 定义模型ID
repo_id = "Qwen/Qwen2-VL-2B-Instruct"

print(f"[-] 开始下载模型: {repo_id}")
print("[-] 如果下载中断，再次运行此脚本即可断点续传。")

try:
    # resume_download=True 开启断点续传
    # local_files_only=False 允许联网
    path = snapshot_download(
        repo_id=repo_id, 
        resume_download=True,
        cache_dir=None # 默认存到 C:\Users\xxx\.cache\huggingface\hub
    )
    print(f"\n[√] 下载成功！模型保存在: {path}")
    print("现在你可以重新运行 pic2ppt_qwen_mirror.py 了。")

except Exception as e:
    print(f"\n[!] 下载失败: {e}")
    print("建议：请手动删除 C:\\Users\\<用户名>\\.cache\\huggingface\\hub 下的相关文件夹后重试。")