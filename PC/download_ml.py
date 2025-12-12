import os
import requests
import tarfile

# 创建存放模型的目录
MODEL_DIR = "models_server"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# PP-OCRv4 Server (服务器版) 模型地址
# 比默认的 Mobile 版大几十倍，精度更高
urls = {
    "det": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar",
    "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar",
    "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
}

def download_and_extract(name, url):
    filename = url.split("/")[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    
    # 1. 下载
    if not os.path.exists(filepath):
        print(f"正在下载 {name} 模型 (可能需要几分钟)...")
        print(f"URL: {url}")
        try:
            response = requests.get(url, stream=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
            print("下载完成。")
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    else:
        print(f"{name} 模型压缩包已存在，跳过下载。")

    # 2. 解压
    print(f"正在解压 {filename}...")
    try:
        with tarfile.open(filepath) as tar:
            tar.extractall(path=MODEL_DIR)
        print("解压完成。\n")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 开始下载 PaddleOCR Server (高精度) 模型 ===")
    success = True
    for name, url in urls.items():
        if not download_and_extract(name, url):
            success = False
    
    if success:
        print("=== 所有模型准备就绪 ===")
        print(f"模型已保存在: {os.path.abspath(MODEL_DIR)}")
    else:
        print("部分模型下载失败，请检查网络。")