const canvas = new fabric.Canvas('c');
let currentFilename = null;
let originalImageWidth = 0;
let originalImageHeight = 0;

// 配置 Fabric
fabric.Object.prototype.transparentCorners = false;
fabric.Object.prototype.cornerColor = 'blue';
fabric.Object.prototype.cornerStyle = 'circle';

// 上传处理
document.getElementById('uploadInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    showLoading("正在上传并进行 OCR 识别 (CPU)...");
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        
        currentFilename = data.filename;
        loadEditor(data);
        
        document.getElementById('genBtn').disabled = false;
        document.getElementById('genBtn').classList.remove('opacity-50', 'cursor-not-allowed');
    } catch (err) {
        alert("处理失败: " + err.message);
    } finally {
        hideLoading();
    }
});

function loadEditor(data) {
    canvas.clear();
    originalImageWidth = data.width;
    originalImageHeight = data.height;
    
    // 设置画布大小
    canvas.setWidth(data.width);
    canvas.setHeight(data.height);

    // 加载背景图
    fabric.Image.fromURL(`/api/image/${data.filename}`, (img) => {
        img.set({ selectable: false });
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
    });

    // 绘制框
    data.boxes.forEach(box => {
        addRect(box.x, box.y, box.w, box.h, box.text);
    });
}

function addRect(left, top, width, height, text) {
    const rect = new fabric.Rect({
        width: width, height: height,
        fill: 'rgba(0,0,0,0)',
        stroke: '#00ff00', strokeWidth: 2,
        originX: 'left', originY: 'top'
    });
    
    const group = new fabric.Group([rect], {
        left: left, top: top,
        width: width, height: height,
        transparentCorners: false
    });
    
    // 自定义属性存储文字
    group.ocrText = text;
    
    canvas.add(group);
    return group;
}

// 选中事件监听
const textEditor = document.getElementById('textEditor');
const boxInfo = document.getElementById('boxInfo');

canvas.on('selection:created', updateSidebar);
canvas.on('selection:updated', updateSidebar);
canvas.on('selection:cleared', () => {
    textEditor.classList.add('hidden');
    boxInfo.innerText = "未选中任何框";
});

function updateSidebar() {
    const activeObj = canvas.getActiveObject();
    if (!activeObj) return;
    
    boxInfo.innerText = `X: ${Math.round(activeObj.left)}, Y: ${Math.round(activeObj.top)}\nW: ${Math.round(activeObj.getScaledWidth())}, H: ${Math.round(activeObj.getScaledHeight())}`;
    
    textEditor.value = activeObj.ocrText || "";
    textEditor.classList.remove('hidden');
}

textEditor.addEventListener('input', (e) => {
    const activeObj = canvas.getActiveObject();
    if (activeObj) {
        activeObj.ocrText = e.target.value;
    }
});

// === 功能函数 ===

window.addBox = function() {
    // 在视图中心添加框
    const center = canvas.getVpCenter();
    const group = addRect(center.x - 75, center.y - 15, 150, 30, "新文本");
    canvas.setActiveObject(group);
    canvas.requestRenderAll();
}

window.deleteActiveBox = function() {
    const activeObj = canvas.getActiveObject();
    if (activeObj) {
        canvas.remove(activeObj);
        canvas.discardActiveObject();
        canvas.requestRenderAll();
    }
}

window.generatePPT = async function() {
    if (!currentFilename) return;
    
    showLoading("正在修复图片并生成 PPT (CPU 较慢请耐心等待)...");
    
    const boxes = [];
    canvas.getObjects().forEach(obj => {
        if (obj.type === 'group') {
            boxes.push({
                x: obj.left,
                y: obj.top,
                w: obj.getScaledWidth(),
                h: obj.getScaledHeight(),
                text: obj.ocrText || ""
            });
        }
    });

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ filename: currentFilename, boxes: boxes })
        });
        
        const data = await res.json();
        if (data.download_url) {
            window.location.href = data.download_url;
        } else {
            throw new Error("生成失败");
        }
    } catch (err) {
        alert("生成出错: " + err.message);
    } finally {
        hideLoading();
    }
}

function showLoading(msg) {
    document.getElementById('loadingText').innerText = msg;
    document.getElementById('loading').classList.remove('hidden');
}
function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// === 键盘事件 (复刻 WASD 微调) ===
document.addEventListener('keydown', (e) => {
    const activeObj = canvas.getActiveObject();
    if (!activeObj) {
        // 全局快捷键
        if (e.ctrlKey && e.key === 'n') { e.preventDefault(); addBox(); }
        return;
    }

    // 如果焦点在文本框，不触发快捷键
    if (document.activeElement === textEditor) return;

    const step = e.shiftKey ? 5 : 1;
    let modified = false;

    // 删除
    if (e.key === 'Delete' || (e.ctrlKey && e.key === 'd')) {
        deleteActiveBox();
        e.preventDefault();
        return;
    }

    // 方向键整体移动
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        if (e.key === 'ArrowUp') activeObj.top -= step;
        if (e.key === 'ArrowDown') activeObj.top += step;
        if (e.key === 'ArrowLeft') activeObj.left -= step;
        if (e.key === 'ArrowRight') activeObj.left += step;
        modified = true;
    }

    // WASD 边缘微调 (模拟 Tkinter 逻辑)
    // 注意：Fabric 的 scaling 机制和 xywh 不同，这里简化为直接改宽高
    // Fabric 默认中心点可能会变，这里需要小心处理
    
    const w = activeObj.getScaledWidth();
    const h = activeObj.getScaledHeight();

    if (e.key.toLowerCase() === 'w') { // 调整上边
        if (e.shiftKey) { // 下移 (height 减小, top 增加)
            if (h > step) { activeObj.set('height', (h - step) / activeObj.scaleY); activeObj.top += step; }
        } else { // 上移 (height 增加, top 减小)
            activeObj.set('height', (h + step) / activeObj.scaleY); activeObj.top -= step;
        }
        modified = true;
    }
    
    if (e.key.toLowerCase() === 's') { // 调整下边
         // S: 下边下移 (增加高度)
         if (!e.shiftKey) {
             activeObj.set('height', (h + step) / activeObj.scaleY);
         } else { // Shift+S: 下边上移 (减小高度)
             if (h > step) activeObj.set('height', (h - step) / activeObj.scaleY);
         }
         modified = true;
    }

    if (e.key.toLowerCase() === 'a') { // 调整左边
        if (!e.shiftKey) { // A: 左边左移 (x 减小, w 增加)
            activeObj.set('width', (w + step) / activeObj.scaleX); activeObj.left -= step;
        } else { // Shift+A: 左边右移 (x 增加, w 减小)
            if (w > step) { activeObj.set('width', (w - step) / activeObj.scaleX); activeObj.left += step; }
        }
        modified = true;
    }

    if (e.key.toLowerCase() === 'd') { // 调整右边
        if (!e.shiftKey) { // D: 右边右移 (w 增加)
            activeObj.set('width', (w + step) / activeObj.scaleX);
        } else { // Shift+D: 右边左移 (w 减小)
            if (w > step) activeObj.set('width', (w - step) / activeObj.scaleX);
        }
        modified = true;
    }

    // Q/E 整体缩放
    if (e.key.toLowerCase() === 'q') {
        activeObj.scale(activeObj.scaleX * (1 - 0.01 * step));
        modified = true;
    }
    if (e.key.toLowerCase() === 'e') {
        activeObj.scale(activeObj.scaleX * (1 + 0.01 * step));
        modified = true;
    }

    if (modified) {
        activeObj.setCoords(); // 更新坐标响应区
        canvas.requestRenderAll();
        updateSidebar();
    }
});