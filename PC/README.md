# Pic2PPT: 智能图片转可编辑 PPT 工具

**Pic2PPT** 是一个基于 AI 的自动化工具，旨在将包含文字的静态图片（如幻灯片截图、扫描件）转换为**可编辑文字**的 PowerPoint (`.pptx`) 文件。

它不仅仅是简单的 OCR，它还利用图像修复技术（Inpainting）擦除原始文字，并在原位生成样式匹配的可编辑文本框，从而实现“从图片到 PPT”的完美还原。

## ✨ 核心功能

1. **智能 OCR 识别**: 集成 PaddleOCR (Server版模型)，精准识别图片中的中英文文本。
2. **AI 图像去字**: 使用 **LaMa (Large Mask Inpainting)** 模型，自动擦除原始图片上的文字，生成干净的背景图。
3. **样式还原**:
* **颜色提取**: 基于 HSV 空间分析，自动提取文字的主色调。
* **粗体检测**: 检测文字是否加粗。
* **字号自适应**: 自动计算文本框大小，确保生成的 PPT 布局不乱。


4. **双模式运行**:
* **全自动模式 (`pic2ppt.py`)**: 批量处理文件夹，一键生成。
* **人工微调模式 (`manual_editor.py`)**: 提供 GUI 界面，允许用户手动调整、增加或删除识别框，确保 100% 准确率。


5. **智能框修正**: 内置“绿框逻辑” (`smart_refine_box`)，自动紧缩 OCR 边界，去除无关背景，提高修复效果。

---

## 📂 项目结构

```text
Pic2PPT/
├── images/                     # [输入] 存放待转换图片的文件夹
├── models_server/              # [核心] 存放 PaddleOCR Server 推理模型
│   ├── ch_PP-OCRv4_det_server_infer/
│   ├── ch_PP-OCRv4_rec_server_infer/
│   └── ch_ppocr_mobile_v2.0_cls_infer/
├── manual_output/              # [中间件] 存放 GUI 编辑后的 JSON 数据
├── pic2ppt.py                  # [入口] 全自动转换脚本
├── pic2ppt_backend.py          # [核心] 后端逻辑封装（供 GUI 调用）
├── manual_editor.py            # [入口] GUI 可视化编辑器
├── result_ultimate_final.pptx  # [输出] 自动模式的结果
└── final_output.pptx           # [输出] 手动模式的最终结果

```

---

## 🛠️ 环境准备与安装

### 1. 硬件要求

* **GPU (推荐)**: 强烈建议使用 NVIDIA 显卡 (CUDA)。代码会自动检测 GPU，LaMa 模型和 OCR Server 模型在 CPU 上运行较慢。

### 2. Python 依赖

请确保安装以下库：

```bash
pip install opencv-python numpy pillow torch torchvision
pip install paddlepaddle-gpu  # 或者 paddlepaddle (CPU版)
pip install paddleocr
pip install python-pptx
pip install simple-lama-inpainting
pip install tk  # 通常 Python 自带，用于 GUI

```

### 3. 模型准备

代码中硬编码了模型路径为 `models_server`。您需要下载 PaddleOCR V4 Server 模型并放置在对应目录下：

* 检测模型 (Det): `ch_PP-OCRv4_det_server_infer`
* 识别模型 (Rec): `ch_PP-OCRv4_rec_server_infer`
* 分类模型 (Cls): `ch_ppocr_mobile_v2.0_cls_infer`

---

## 🚀 使用指南

### 方式一：全自动转换 (CLI)

适用于对精度要求不苛刻，需要快速批量处理大量图片的场景。

1. 将图片放入 `images/` 文件夹。
2. 运行脚本：
```bash
python pic2ppt.py

```


3. 程序将依次执行 OCR -> 样式分析 -> 背景修复 -> PPT 生成。
4. 结果保存在 `result_ultimate_final.pptx`。

### 方式二：交互式微调 (GUI) **(推荐)**

适用于需要高质量输出的场景。您可以先可视化检查识别结果，修正错位或多余的框。

1. 将图片放入 `images/` 文件夹。
2. 运行编辑器：
```bash
python manual_editor.py

```


3. **在 GUI 中操作**：
* 程序会自动加载第一张图并显示 OCR 结果。
* 使用鼠标点击选择框，使用键盘微调（见下方快捷键）。
* 点击 **"保存当前框 (JSON)"** 保存当前图片的调整结果。
* 切换到下一张图片继续处理。


4. 全部调整完毕后，点击右侧的 **"生成最终 PPT"** 按钮。
5. 结果保存在 `final_output.pptx`。

---

## 🎮 GUI 编辑器快捷键说明

编辑器专为高效率像素级微调设计，支持丰富的键盘操作：

| 功能分类 | 按键 | 作用 |
| --- | --- | --- |
| **选择** | `鼠标左键` | 点击选中文字框 |
|  | `Z` / `X` | 切换上一个 / 下一个选中框 |
| **整体移动** | `↑` `↓` `←` `→` | 移动 1 像素 |
|  | `Shift` + `↑/↓/←/→` | 快速移动 5 像素 |
| **边缘微调** | `W` / `S` | 调整**上/下**边缘 (W:上移, S:下移) |
| (WASD) | `A` / `D` | 调整**左/右**边缘 (A:左移, D:右移) |
|  | `Shift` + `W/S/A/D` | 反向/快速调整边缘 (例如 Shift+W 为上边缘下移) |
| **缩放** | `Q` / `E` | 整体缩小 / 放大 (1px) |
|  | `R` / `T` | 快速缩小 / 放大 (5px) |
| **编辑** | `Ctrl` + `N` | 在上次点击位置**新增**文本框 |
|  | `Ctrl` + `D` / `Del` | **删除**当前选中的框 |
| **导航** | 界面按钮 | 上一张 / 下一张 / 保存 |

---

## ⚙️ 核心参数配置

您可以在 `pic2ppt.py` 或 `pic2ppt_backend.py` 顶部修改以下常量以适应不同需求：

* `MIN_SCORE = 0.75`: OCR 置信度阈值，低于此分数的文字会被当作噪点忽略。
* `FONT_SIZE_FACTOR = 0.98`: 字体大小系数，用于微调 PPT 内文字填满框的程度。
* `MASK_PADDING_PIXELS`: 修复遮罩的膨胀像素，数值越大，擦除范围越大。
* `PADDING_PIXELS_GREEN`: 绿框（最终文本框）的外扩像素。

---

## 🧠 技术原理 (Logic Flow)

1. **预处理**: 遍历图片，PaddleOCR 提取原始坐标（红框）和文本。
2. **噪点过滤**: `is_noise_or_icon` 函数根据长宽比、字符内容（是否包含中文/字母）过滤掉图标和非文字噪点。
3. **智能精修 (Smart Refine)**:
* OCR 的原始框通常偏大。程序在原始框范围内进行二值化和轮廓查找，计算出紧贴文字的最小外接矩形（绿框）。
* **作用**: 绿框用于 PPT 文本框定位，确保文字排版紧凑；红框+外扩用于 LaMa 修复，确保背景擦除干净。


4. **样式分析**:
* 在 HSV 空间滤除背景色（过亮/过暗），提取剩余像素中**饱和度最高**的颜色作为文字颜色。
* 计算二值化后的像素占比，判断是否为**粗体**。


5. **Inpainting (修复)**: 生成黑白 Mask（白色为文字区域），输入 SimpleLama 模型，生成不含文字的纯净背景图。
6. **PPT 组装**:
* 将修复后的背景图设为 Slide 背景。
* 在绿框坐标处插入 Textbox。
* 填入文字，应用分析出的颜色、粗体和计算出的字号。



---

## ⚠️ 注意事项

* **字体**: 程序默认使用 "微软雅黑"，请确保系统已安装该字体。
* **手动保存**: 在 GUI 模式下，**必须点击“保存当前框”** 才会生成 JSON 文件，直接点击“生成 PPT”只会处理已保存 JSON 的图片。
* **LaMa 错误**: 如果遇到 LaMa 报错，程序会降级使用原始图片作为背景（文字不会被擦除，但新文字会覆盖上去）。