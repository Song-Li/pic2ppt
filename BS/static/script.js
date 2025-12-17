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
        // [修正1] 路径从 /api/upload 改为 /upload
        const res = await fetch('/upload', { method: 'POST', body: formData });
        
        if (!res.ok) throw new Error(`上传失败: ${res.status}`);

        const responseJson = await res.json();
        
        // [修正2] 后端返回的是 {status: "ok", data: {...}}，所以要取 responseJson.data
        const data = responseJson.data;

        if (!data) throw new Error("后端返回数据格式错误");
        
        currentFilename = data.filename;
        loadEditor(data);
        
        document.getElementById('genBtn').disabled = false;
        document.getElementById('genBtn').classList.remove('opacity-50', 'cursor-not-allowed');
    } catch (err) {
        console.error(err);
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

    // [修正3] 图片路径改为 /static/文件名 (或者直接用后端返回的 data.url)
    const imageUrl = `/static/${data.filename}`;

    fabric.Image.fromURL(imageUrl, (img) => {
        if (!img) {
            alert("无法加载图片，请检查 static 文件夹中是否存在该文件");
            return;
        }
        img.set({ selectable: false });
        canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
    });

    // 绘制框
    if (data.boxes) {
        data.boxes.forEach(box => {
            // 注意：检查后端返回的字段名是否也是 x, y, w, h，如果是 raw_x 等需调整
            // 假设 core.py 返回的结构里包含 x, y, w, h
            // 这里根据 core.py 的逻辑，initial_boxes 是个列表，需要确认 key
            // 暂时假设后端传回来的 boxes 是对象数组。如果 core.py 传回的是列表数组，这里可能还会报错。
            // 我们先按对象处理，如果报错再调整。
             addRect(box.x, box.y, box.w, box.h, box.text);
        });
    }
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
                // 给后端传参时需要 id，这里简单给个 0，或者后端若不强制校验 id 可忽略
                id: 0, 
                x: Math.round(obj.left),
                y: Math.round(obj.top),
                w: Math.round(obj.getScaledWidth()),
                h: Math.round(obj.getScaledHeight()),
                text: obj.ocrText || "",
                color: [0, 0, 0], // 默认黑色
                is_bold: false
            });
        }
    });

    try {
        // [修正4] 路径从 /api/generate 改为 /generate
        const res = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ filename: currentFilename, boxes: boxes })
        });
        
        if (!res.ok) throw new Error(`生成失败: ${res.status}`);

        const responseJson = await res.json();
        
        // 这里的 responseJson 可能是 {status: "ok", ppt_url: "..."}
        if (responseJson.ppt_url) {
            window.location.href = responseJson.ppt_url;
        } else {
            throw new Error("生成返回数据异常");
        }
    } catch (err) {
        alert("生成出错: " + err.message);
    } finally {
        hideLoading();
    }
}

function showLoading(msg) {
    const loadingText = document.getElementById('loadingText');
    const loading = document.getElementById('loading');
    if (loadingText) loadingText.innerText = msg;
    if (loading) loading.classList.remove('hidden');
}
function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) loading.classList.add('hidden');
}

// === 键盘事件 ===
document.addEventListener('keydown', (e) => {
    const activeObj = canvas.getActiveObject();
    if (!activeObj) {
        if (e.ctrlKey && e.key === 'n') { e.preventDefault(); addBox(); }
        return;
    }

    if (document.activeElement === textEditor) return;

    const step = e.shiftKey ? 5 : 1;
    let modified = false;

    if (e.key === 'Delete' || (e.ctrlKey && e.key === 'd')) {
        deleteActiveBox();
        e.preventDefault();
        return;
    }

    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
        if (e.key === 'ArrowUp') activeObj.top -= step;
        if (e.key === 'ArrowDown') activeObj.top += step;
        if (e.key === 'ArrowLeft') activeObj.left -= step;
        if (e.key === 'ArrowRight') activeObj.left += step;
        modified = true;
    }

    const w = activeObj.getScaledWidth();
    const h = activeObj.getScaledHeight();

    if (e.key.toLowerCase() === 'w') {
        if (e.shiftKey) { 
            if (h > step) { activeObj.set('height', (h - step) / activeObj.scaleY); activeObj.top += step; }
        } else {
            activeObj.set('height', (h + step) / activeObj.scaleY); activeObj.top -= step;
        }
        modified = true;
    }
    
    if (e.key.toLowerCase() === 's') {
         if (!e.shiftKey) {
             activeObj.set('height', (h + step) / activeObj.scaleY);
         } else {
             if (h > step) activeObj.set('height', (h - step) / activeObj.scaleY);
         }
         modified = true;
    }

    if (e.key.toLowerCase() === 'a') {
        if (!e.shiftKey) { 
            activeObj.set('width', (w + step) / activeObj.scaleX); activeObj.left -= step;
        } else { 
            if (w > step) { activeObj.set('width', (w - step) / activeObj.scaleX); activeObj.left += step; }
        }
        modified = true;
    }

    if (e.key.toLowerCase() === 'd') {
        if (!e.shiftKey) { 
            activeObj.set('width', (w + step) / activeObj.scaleX);
        } else { 
            if (w > step) activeObj.set('width', (w - step) / activeObj.scaleX);
        }
        modified = true;
    }

    if (e.key.toLowerCase() === 'q') {
        activeObj.scale(activeObj.scaleX * (1 - 0.01 * step));
        modified = true;
    }
    if (e.key.toLowerCase() === 'e') {
        activeObj.scale(activeObj.scaleX * (1 + 0.01 * step));
        modified = true;
    }

    if (modified) {
        activeObj.setCoords();
        canvas.requestRenderAll();
        updateSidebar();
    }
});
