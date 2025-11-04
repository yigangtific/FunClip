# 使用轻量级Python镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖（使用国内源加速）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露应用端口（与代码中保持一致）
EXPOSE 7860

# 启动命令
CMD ["python", "funclip:launch_final.py"]
