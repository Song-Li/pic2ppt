import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from .core import ImageProcessor

app = FastAPI()

# 静态资源目录
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化处理核心
processor = ImageProcessor()

class BoxItem(BaseModel):
    id: int
    x: int; y: int; w: int; h: int
    text: str
    color: List[int]
    is_bold: bool

class GenRequest(BaseModel):
    filename: str
    boxes: List[BoxItem]

@app.post("/upload")
async def api_upload(file: UploadFile = File(...)):
    try:
        # 保存图片
        ext = os.path.splitext(file.filename)[1]
        fname = f"{uuid.uuid4()}{ext}"
        fpath = f"static/{fname}"
        with open(fpath, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 核心 OCR 分析
        data = processor.process_image(fpath)
        data['filename'] = fname
        data['url'] = f"/static/{fname}"
        return {"status": "ok", "data": data}
        
    except Exception as e:
        print(e)
        raise HTTPException(500, detail=str(e))

@app.post("/generate")
async def api_generate(req: GenRequest):
    try:
        fpath = f"static/{req.filename}"
        if not os.path.exists(fpath): raise HTTPException(404, "图片不存在")
        
        out_name = f"ppt_{uuid.uuid4()}.pptx"
        out_path = f"static/{out_name}"
        
        # 转换数据格式
        boxes = [b.dict() for b in req.boxes]
        
        # 核心 PPT 生成
        processor.generate_ppt(fpath, boxes, out_path)
        
        return {"status": "ok", "ppt_url": f"/static/{out_name}"}
        
    except Exception as e:
        print(e)
        raise HTTPException(500, detail=str(e))

@app.get("/")
async def read_index():
    # 确保 static/index.html 存在
    if os.path.exists("static/index.html"):
        return FileResponse('static/index.html')

@app.get("/style.css")
async def read_style():
    if os.path.exists("static/style.css"):
        return FileResponse('static/style.css')

@app.get("/script.js")
async def read_script():
    if os.path.exists("static/script.js"):
        return FileResponse('static/script.js')
    return {"error": "static/index.html not found"}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 允许外部访问
    uvicorn.run(app, host="0.0.0.0", port=8000)
