import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import json
import os
import glob
from pic2ppt_backend import ImageToPPTConverter 

# 目标输出文件夹
MANUAL_OUTPUT_DIR = "manual_output"
FINAL_PPT_NAME = "final_output.pptx"

class BoxEditorApp:
    def __init__(self, master, converter):
        self.master = master
        master.title("Pic2PPT 手动边界编辑器")
        
        self.converter = converter
        self.image_files = []
        self.current_image_index = -1
        self.current_boxes = []     
        self.selected_box_index = -1
        self.zoom_factor = 1.0 
        self.scale_factor = 1.0 
        self.last_click_coords = (0, 0)
        
        self.setup_ui()
        self.load_folder()
        self.bind_keys()

    def load_folder(self, initial_folder="images"):
        """加载图片文件夹"""
        self.image_files = sorted(glob.glob(os.path.join(initial_folder, "*.[jpg][png][jpeg]*")))
        if self.image_files:
            self.current_image_index = 0
            self.load_image()
        else:
            self.info_label.config(text="错误: 未找到图片，请创建 'images' 文件夹。")

    def setup_ui(self):
        main_pane = ttk.Panedwindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # === 左侧：图像画布 ===
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)

        # 1. 顶部控制条
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.prev_button = ttk.Button(control_frame, text="<< 上一张", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(control_frame, text="下一张 >>", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="保存当前框 (JSON)", command=self.save_boxes)
        self.save_button.pack(side=tk.LEFT, padx=15)
        
        self.info_label = ttk.Label(control_frame, text="状态: 未加载图片")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # 2. 选中框信息
        self.box_info_label = ttk.Label(left_frame, text="选中框: None")
        self.box_info_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # 3. Canvas
        self.canvas = tk.Canvas(left_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.select_box_by_click)
        self.canvas.focus_set() # 确保 canvas 具有键盘焦点

        # === 右侧：引导与功能区 ===
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # 1. 引导信息
        self.create_guide_panel(right_frame)

        # 2. 功能按钮
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="新增框 (Ctrl+N)", command=self.add_new_box).pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="删除选中框 (Ctrl+D)", command=self.delete_selected_box).pack(fill=tk.X, pady=3)

        # 3. 生成 PPT 按钮
        self.generate_ppt_button = ttk.Button(right_frame, text="生成最终 PPT", command=self.generate_final_ppt)
        self.generate_ppt_button.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)


    def create_guide_panel(self, parent_frame):
        """创建操作引导面板"""
        guide_frame = ttk.LabelFrame(parent_frame, text="操作引导 (像素级微调)")
        guide_frame.pack(fill=tk.X, padx=10, pady=10)

        guide_text = """
--- 整体操作 ---
方向键 (↑↓←→): 整体移动选中框 (1px)
Shift + 方向键: 快速整体移动 (5px)
X / Z 键: 切换选中框

--- [新增] 增删功能 ---
Ctrl+N: 在上次点击位置新增默认框
Ctrl+D / Delete: 删除选中框

--- 边缘微调 (WASD) ---
W / S / A / D: 边缘微调 (1px)
Shift+W / Shift+S / ...: 快速微调 (5px)

    W: 上边上移 (缩小框高)
    Shift+W: 上边下移 (放大框高)
    S: 下边下移 (放大框高)
    Shift+S: 下边上移 (缩小框高)

--- 整体缩放 (QERT) ---
Q / E: 整体缩小 / 放大 (1px)
R / T: 快速缩小 / 放大 (5px)
        """
        guide_label = ttk.Label(guide_frame, text=guide_text, justify=tk.LEFT, font=('Arial', 10, 'bold'))
        guide_label.pack(padx=10, pady=5)
        
    def bind_keys(self):
        # 修复 Shift + WASD 绑定，使用大写字母区分
        # W: 上边上移 (-1) | Shift+W: 上边下移 (+1)
        self.master.bind("w", lambda event: self.move_edge('top', -1))
        self.master.bind("W", lambda event: self.move_edge('top', 1)) 
        
        # S: 下边下移 (+1) | Shift+S: 下边上移 (-1)
        self.master.bind("s", lambda event: self.move_edge('bottom', 1))
        self.master.bind("S", lambda event: self.move_edge('bottom', -1)) 
        
        # A: 左侧左移 (-1) | Shift+A: 左侧右移 (+1)
        self.master.bind("a", lambda event: self.move_edge('left', -1))
        self.master.bind("A", lambda event: self.move_edge('left', 1)) 
        
        # D: 右侧右移 (+1) | Shift+D: 右侧左移 (-1)
        self.master.bind("d", lambda event: self.move_edge('right', 1))
        self.master.bind("D", lambda event: self.move_edge('right', -1)) 
        
        # WASD 快速移动 (5px)
        self.master.bind("<Shift-w>", lambda event: self.move_edge('top', -5))
        self.master.bind("<Shift-s>", lambda event: self.move_edge('bottom', 5))
        self.master.bind("<Shift-a>", lambda event: self.move_edge('left', -5))
        self.master.bind("<Shift-d>", lambda event: self.move_edge('right', 5))
        
        # 方向键 整体移动
        self.master.bind("<Up>", lambda event: self.move_box(0, -1))
        self.master.bind("<Shift-Up>", lambda event: self.move_box(0, -5))
        self.master.bind("<Down>", lambda event: self.move_box(0, 1))
        self.master.bind("<Shift-Down>", lambda event: self.move_box(0, 5))
        self.master.bind("<Left>", lambda event: self.move_box(-1, 0))
        self.master.bind("<Shift-Left>", lambda event: self.move_box(-5, 0))
        self.master.bind("<Right>", lambda event: self.move_box(1, 0))
        self.master.bind("<Shift-Right>", lambda event: self.move_box(5, 0))

        # QERT 整体放大/缩小
        self.master.bind("q", lambda event: self.scale_box_uniform(-1))
        self.master.bind("e", lambda event: self.scale_box_uniform(1))
        self.master.bind("r", lambda event: self.scale_box_uniform(-5))
        self.master.bind("t", lambda event: self.scale_box_uniform(5))

        # Z/X 切换选中框
        self.master.bind("z", lambda event: self.change_selected_box(-1))
        self.master.bind("x", lambda event: self.change_selected_box(1))
        
        # 增删功能 (Ctrl 绑定)
        self.master.bind("<Control-n>", lambda event: self.add_new_box())
        self.master.bind("<Control-d>", lambda event: self.delete_selected_box())
        self.master.bind("<Delete>", lambda event: self.delete_selected_box())
        self.master.bind("<BackSpace>", lambda event: self.delete_selected_box())

    def load_image(self):
        if not self.image_files: return
        
        path = self.image_files[self.current_image_index]
        self.current_image_path = path
        
        self.original_image = Image.open(path)
        self.image_width, self.image_height = self.original_image.size
        
        json_filename = os.path.splitext(os.path.basename(path))[0] + "_manual.json"
        json_path = os.path.join(MANUAL_OUTPUT_DIR, json_filename)

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.current_boxes = [[b['x'], b['y'], b['w'], b['h'], b['text']] for b in data['boxes']]
            print(f"Loaded manual data for {os.path.basename(path)}")
        else:
            initial_data = self.converter.get_initial_boxes(path)
            self.current_boxes = [[d[0], d[1], d[2], d[3], d[4]] for d in initial_data]


        self.selected_box_index = 0 if self.current_boxes else -1
        
        self.update_canvas_display()
        self.info_label.config(text=f"图片 {self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(path)}")

    def update_canvas_display(self):
        if not hasattr(self, 'original_image'): return

        self.canvas.delete("all")
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        scale_w = canvas_w / self.image_width
        scale_h = canvas_h / self.image_height
        self.scale_factor = min(scale_w, scale_h) * self.zoom_factor 
        
        new_w = int(self.image_width * self.scale_factor)
        new_h = int(self.image_height * self.scale_factor)

        if new_w <= 0 or new_h <= 0: return

        resized_img = self.original_image.resize((new_w, new_h))
        draw = ImageDraw.Draw(resized_img)
        
        for i, box in enumerate(self.current_boxes):
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            draw_x1 = int(x * self.scale_factor)
            draw_y1 = int(y * self.scale_factor)
            draw_x2 = int((x + w) * self.scale_factor)
            draw_y2 = int((y + h) * self.scale_factor)
            
            color = "blue"
            width = 1
            if i == self.selected_box_index:
                color = "green" 
                width = 2
            
            draw.rectangle([draw_x1, draw_y1, draw_x2, draw_y2], outline=color, width=width)
            
            text = box[4]
            draw.text((draw_x1, draw_y1 - 15), text[:10], fill=color)

        self.tk_image = ImageTk.PhotoImage(resized_img)
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_image, anchor=tk.CENTER)
        
        self.update_box_info()

    def update_box_info(self):
        if self.selected_box_index != -1 and self.current_boxes:
            box = self.current_boxes[self.selected_box_index]
            x, y, w, h = box[0], box[1], box[2], box[3]
            text = box[4]
            self.box_info_label.config(text=f"选中框 #{self.selected_box_index + 1} ({text[:15]}...): X:{x}, Y:{y}, W:{w}, H:{h}")
        else:
            self.box_info_label.config(text="选中框: None")

    def select_box_by_click(self, event):
        if not self.current_boxes: return
        
        click_x_canvas = event.x
        click_y_canvas = event.y
        
        image_x = int(click_x_canvas / self.scale_factor)
        image_y = int(click_y_canvas / self.scale_factor)
        
        self.last_click_coords = (image_x, image_y) 

        min_distance = float('inf')
        closest_index = -1
        
        for i, box in enumerate(self.current_boxes):
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            if x <= image_x <= x + w and y <= image_y <= y + h:
                center_x, center_y = x + w / 2, y + h / 2
                distance = (image_x - center_x)**2 + (image_y - center_y)**2
                
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i

        if closest_index != -1:
            self.selected_box_index = closest_index
            self.update_canvas_display()

    def move_box(self, dx, dy):
        if self.selected_box_index == -1: return
        
        box = self.current_boxes[self.selected_box_index]
        box[0] = max(0, box[0] + dx) 
        box[1] = max(0, box[1] + dy) 
        
        self.update_canvas_display()

    def move_edge(self, edge, delta):
        """调整选中框的边缘 (WASD + Shift)"""
        if self.selected_box_index == -1: return
        
        box = self.current_boxes[self.selected_box_index]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        min_dim = 1 

        if edge == 'top':
            if y + delta >= 0 and h - delta >= min_dim:
                box[1] += delta
                box[3] -= delta
        elif edge == 'bottom':
            # delta > 0: 下移 (放大)； delta < 0: 上移 (缩小)
            if y + h + delta <= self.image_height and h + delta >= min_dim:
                box[3] += delta
        elif edge == 'left':
            if x + delta >= 0 and w - delta >= min_dim:
                box[0] += delta
                box[2] -= delta
        elif edge == 'right':
            # delta > 0: 右移 (放大)； delta < 0: 左移 (缩小)
            if x + w + delta <= self.image_width and w + delta >= min_dim:
                box[2] += delta

        self.update_canvas_display()

    def scale_box_uniform(self, delta):
        if self.selected_box_index == -1: return

        box = self.current_boxes[self.selected_box_index]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        new_x = max(0, x - delta)
        new_y = max(0, y - delta)
        new_w = w + 2 * delta
        new_h = h + 2 * delta
        
        if new_w <= 0 or new_h <= 0:
            print("边界尺寸过小，无法继续缩小。")
            return
            
        if new_x + new_w > self.image_width or new_y + new_h > self.image_height:
            print("边界已达到图像边缘，无法继续放大。")
            return

        box[0], box[1], box[2], box[3] = new_x, new_y, new_w, new_h
        
        self.update_canvas_display()

    def add_new_box(self):
        """在上次点击位置添加一个默认框 (Ctrl+N)"""
        if not hasattr(self, 'original_image'): 
            messagebox.showwarning("警告", "请先加载图片。")
            return

        x_click, y_click = self.last_click_coords
        
        default_w, default_h = 150, 30
        
        x = max(0, min(x_click - default_w // 2, self.image_width - default_w))
        y = max(0, min(y_click - default_h // 2, self.image_height - default_h))
        
        new_box = [x, y, default_w, default_h, "手动添加文本"]
        
        self.current_boxes.append(new_box)
        self.selected_box_index = len(self.current_boxes) - 1
        
        self.update_canvas_display()
        print("已在点击位置附近新增一个默认框。")

    def delete_selected_box(self):
        """删除当前选中的框 (Ctrl+D / Delete / Backspace)"""
        if self.selected_box_index == -1 or not self.current_boxes:
            print("没有选中框可删除。")
            return
            
        del self.current_boxes[self.selected_box_index]
        
        if self.current_boxes:
            self.selected_box_index = min(self.selected_box_index, len(self.current_boxes) - 1)
        else:
            self.selected_box_index = -1
            
        self.update_canvas_display()
        print("已删除选中的框。")
    
    def change_selected_box(self, delta):
        if not self.current_boxes: return
        
        new_index = (self.selected_box_index + delta) % len(self.current_boxes)
        self.selected_box_index = new_index
        self.update_canvas_display()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()
            
    def save_boxes(self):
        if not self.current_boxes:
            print("没有可保存的框。")
            return
            
        filename = os.path.basename(self.current_image_path)
        
        if not os.path.exists(MANUAL_OUTPUT_DIR):
            os.makedirs(MANUAL_OUTPUT_DIR)
            
        output_filename = os.path.splitext(filename)[0] + "_manual.json"
        output_path = os.path.join(MANUAL_OUTPUT_DIR, output_filename)

        output_data = {
            "image_path": self.current_image_path,
            "boxes": []
        }
        
        for box in self.current_boxes:
            output_data["boxes"].append({
                "x": box[0],
                "y": box[1],
                "w": box[2],
                "h": box[3],
                "text": box[4]
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        messagebox.showinfo("保存成功", f"框已保存到: {output_path}")

    def generate_final_ppt(self):
        if not os.path.exists(MANUAL_OUTPUT_DIR):
            messagebox.showerror("错误", "请先保存至少一张图片的手动调整框！")
            return
            
        try:
            all_image_files = sorted(glob.glob(os.path.join("images", "*.[jpg][png][jpeg]*")))
            
            self.converter.generate_ppt_from_manual_data(all_image_files, FINAL_PPT_NAME, MANUAL_OUTPUT_DIR)
            
            messagebox.showinfo("生成完成", f"最终 PPT 已生成：{FINAL_PPT_NAME}")
        except Exception as e:
            messagebox.showerror("生成失败", f"生成 PPT 时发生错误: {e}")


if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")
        print("请在 'images' 文件夹中放入图片。")
        
    if not os.path.exists("models_server"):
        print("\n!!! 警告：Server 模型未找到。请先运行 download_models.py !!!\n")

    try:
        converter_instance = ImageToPPTConverter() 
    except FileNotFoundError as e:
        print(e)
        input("按任意键退出...")
        sys.exit()

    root = tk.Tk()
    app = BoxEditorApp(root, converter_instance)
    
    root.geometry("1200x800")
    root.bind("<Configure>", lambda event: app.update_canvas_display() if event.widget == root else None)

    root.mainloop()