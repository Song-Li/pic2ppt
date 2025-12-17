import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
def debug_upload(file: UploadFile = File(...)):
    print(f"✅ 成功接收到文件: {file.filename}")
    return {"status": "ok", "filename": file.filename}

if __name__ == "__main__":
    print("🚀 启动调试服务器: http://0.0.0.0:58000")
    uvicorn.run(app, host="0.0.0.0", port=58000)