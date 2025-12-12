import os
import cv2
import numpy as np
from PIL import Image
import torch
from simple_lama_inpainting import SimpleLama
from paddleocr import PaddleOCR
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_VERTICAL_ANCHOR

# ================= 配置 =================
# 强制使用 CPU
DEVICE = "cpu"
# PPT/排版配置
FONT_SIZE_FACTOR = 0.95
MASK_DILATE_ITER = 2
MASK_PADDING_PIXELS = 2
PADDING_PIXELS_GREEN = 2

class ImageProcessor:
    def __init__(self):
        print("[-] 初始化 PaddleOCR (Mobile模型, CPU模式, 优化高分辨率)...")
        # 使用默认的 Mobile 模型，自动下载。
        # det_limit_side_len 设置为 1960 或更高以支持高清图识别
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=False,
            show_log=False,
            det_limit_side_len=1960 
        )
        
        print("[-] 初始化 LaMa Inpainting (CPU模式)...")
        self.lama = SimpleLama()
        # 强制 LaMa 使用 CPU
        self.lama.device = torch.device('cpu') 
        if hasattr(self.lama, 'model'):
            self.lama.model.to(self.lama.device)
            
        print("[√] 模型初始化完成")

    def smart_refine_box(self, img, x, y, w, h):
        """绿框逻辑：返回精准贴合文字的尺寸"""
        if w <= 0 or h <= 0: return x, y, w, h
        pad_probe = 2 
        H, W = img.shape[:2]
        y1, y2 = max(0, y - pad_probe), min(H, y + h + pad_probe)
        x1, x2 = max(0, x - pad_probe), min(W, x + w + pad_probe)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return x, y, w, h

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        points = cv2.findNonZero(binary)
        
        if points is not None:
            rx, ry, rw, rh = cv2.boundingRect(points)
            real_x = x1 + rx
            real_y = y1 + ry
            final_x = max(0, real_x - PADDING_PIXELS_GREEN)
            final_y = max(0, real_y - PADDING_PIXELS_GREEN)
            final_w = min(W - final_x, rw + 2*PADDING_PIXELS_GREEN)
            final_h = min(H - final_y, rh + 2*PADDING_PIXELS_GREEN)
            return int(final_x), int(final_y), int(final_w), int(final_h)
        return x, y, w, h

    def analyze_style(self, img, box):
        """颜色和粗体分析"""
        x, y, w, h = map(int, box)
        if w <= 0 or h <= 0: return (0,0,0), False
        roi = img[y:y+h, x:x+w]
        if roi.size == 0: return (0,0,0), False

        is_bold = False
        try:
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 50])
            upper = np.array([179, 255, 200])
            mask = cv2.inRange(roi_hsv, lower, upper)
            valid_pixels = roi[mask != 0]
            
            if valid_pixels.size > 0:
                # 简单取均值，或者你可以保留之前的复杂逻辑
                color_bgr = cv2.mean(valid_pixels)[:3]
            else:
                color_bgr = cv2.mean(roi)[:3]

            text_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if (cv2.countNonZero(binary) / binary.size) > 0.45:
                is_bold = True
        except:
            text_color = (0,0,0)
        return text_color, is_bold

    def process_ocr(self, img_path):
        """执行 OCR 并返回 Web 端可用的 JSON 数据"""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return []

        # PaddleOCR 运行
        result = self.ocr.ocr(img_bgr, cls=True)
        lines = result[0] if (result and result[0]) else []
        
        boxes_data = []
        for idx, line in enumerate(lines):
            coords = line[0]
            text = line[1][0]
            score = line[1][1]
            
            if score < 0.60: continue # 适当降低阈值适应小模型

            coords_np = np.array(coords, dtype=np.int32)
            raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(coords_np)

            # 过滤极小噪点
            if raw_w < 5 or raw_h < 5: continue

            # 计算 Refine 框
            g_x, g_y, g_rw, g_rh = self.smart_refine_box(img_bgr, raw_x, raw_y, raw_w, raw_h)

            boxes_data.append({
                "id": idx,
                "x": g_x, "y": g_y, "w": g_rw, "h": g_rh,
                "text": text
            })
        
        return {
            "width": img_bgr.shape[1],
            "height": img_bgr.shape[0],
            "boxes": boxes_data
        }

    def generate_ppt(self, img_path, boxes_data, output_path):
        """根据前端调整后的数据生成 PPT"""
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        img_bgr = cv2.imread(img_path)
        h_img, w_img = img_bgr.shape[:2]
        
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        valid_blocks = []

        # 1. 准备数据和 Mask
        for block in boxes_data:
            x, y, w, h = int(block['x']), int(block['y']), int(block['w']), int(block['h'])
            text = block['text']
            
            # 重新分析颜色（因为可能框变了）
            color, is_bold = self.analyze_style(img_bgr, (x, y, w, h))
            
            valid_blocks.append({
                "text": text,
                "rect": (x, y, w, h),
                "color": color,
                "bold": is_bold
            })

            # 绘制 Mask
            m_x1 = max(0, x - MASK_PADDING_PIXELS)
            m_y1 = max(0, y - MASK_PADDING_PIXELS)
            m_x2 = min(w_img, x + w + MASK_PADDING_PIXELS)
            m_y2 = min(h_img, y + h + MASK_PADDING_PIXELS)
            cv2.rectangle(mask, (m_x1, m_y1), (m_x2, m_y2), 255, -1)

        # 2. LaMa 修复 (CPU)
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=MASK_DILATE_ITER)
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            # 必须确保在 CPU 上运行
            clean_pil = self.lama(Image.fromarray(img_rgb), Image.fromarray(mask_dilated))
            clean_bgr = cv2.cvtColor(np.array(clean_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"LaMa Error: {e}")
            clean_bgr = img_bgr

        # 3. 生成 PPT Slide
        temp_bg = output_path + ".temp.png"
        cv2.imwrite(temp_bg, clean_bgr)

        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(temp_bg, 0, 0, width=prs.slide_width, height=prs.slide_height)

        scale_x = prs.slide_width / w_img
        scale_y = prs.slide_height / h_img

        for block in valid_blocks:
            x, y, rw, rh = block['rect']
            textbox = slide.shapes.add_textbox(
                int(x * scale_x), int(y * scale_y),
                int(rw * scale_x), int(rh * scale_y)
            )
            tf = textbox.text_frame
            tf.word_wrap = False
            tf.vertical_anchor = MSO_VERTICAL_ANCHOR.MIDDLE
            tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
            
            p = tf.paragraphs[0]
            p.text = block['text']
            p.font.name = "Microsoft YaHei" # 兼容性更好的字体名
            p.font.color.rgb = RGBColor(*block['color'])
            # p.font.bold = block['bold']
            
            # 字号计算
            H_pt = (rh / h_img) * (prs.slide_height.inches * 72)
            char_count = len(block['text'])
            W_pt = (rw / w_img) * (prs.slide_width.inches * 72)
            W_pt_constraint = W_pt / (char_count if char_count > 0 else 1) / 1.0
            final_pt_size = min(H_pt, W_pt_constraint)
            p.font.size = Pt(max(9, final_pt_size * FONT_SIZE_FACTOR))

        prs.save(output_path)
        if os.path.exists(temp_bg): os.remove(temp_bg)
        return output_path