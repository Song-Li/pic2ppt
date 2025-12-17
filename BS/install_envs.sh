# 1. 创建干净的 Python 3.10 环境
conda create -n pic_prod python=3.10 -y
conda activate pic_prod

# 2. 安装 PyTorch (GPU版, 假设你的 CUDA 是 11.8 或 12.x，这里用 pytorch 官方源自动匹配)
# 这步最关键，conda 会自动帮你把 cudatoolkit 依赖装好
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装 PaddlePaddle (GPU版)
# 百度官方推荐用 pip 安装 GPU 版，这比 conda 里的更新更及时
# 对应 CUDA 11.8 (绝大多数显卡通用)
python -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 4. 安装 PaddleOCR 和 LaMa 以及 Web 框架
# 使用阿里源加速，同时强制安装 protobuf 3.20.x 以防版本冲突
pip install "paddleocr>=2.7.0" "protobuf==3.20.3" simple-lama-inpainting fastapi uvicorn python-multipart python-pptx gunicorn opencv-python -i https://mirrors.aliyun.com/pypi/simple/
