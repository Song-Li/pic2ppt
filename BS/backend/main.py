from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import json
from pydantic import BaseModel
from typing import List
from .core import ImageProcessor

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化处理器
processor = ImageProcessor()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# 定义数据模型
class BoxItem(BaseModel):
    x: float
    y: float
    w: float
    h: float
    text: str

class GenerateRequest(BaseModel):
    filename: str
    boxes: List[BoxItem]

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传图片并执行 OCR"""
    try:
        # 保存图片
        file_ext = file.filename.split('.')[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(TEMP_DIR, unique_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 执行 OCR
        result = processor.process_ocr(file_path)
        result['filename'] = unique_name # 返回文件名供后续引用
        
        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """回显图片"""
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

@app.post("/api/generate")
async def generate_ppt(req: GenerateRequest):
    """接收修正后的框，生成 PPT"""
    img_path = os.path.join(TEMP_DIR, req.filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image expired or not found")
    
    output_filename = f"ppt_{req.filename}.pptx"
    output_path = os.path.join(TEMP_DIR, output_filename)
    
    try:
        # 转换 Pydantic model 到 dict list
        boxes_dict = [b.dict() for b in req.boxes]
        processor.generate_ppt(img_path, boxes_dict, output_path)
        
        return JSONResponse(content={"download_url": f"/api/download/{output_filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_ppt(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

# 挂载静态文件 (必须在 API 路由之后)
app.mount("/", StaticFiles(directory="static", html=True), name="static")