# Video_audit
# Video_audit

视频审计


# 构建
docker build -t video-audit:latest .

# 运行 qa（配置通过 --env-file 注入，环境变量优先级高于 .env 文件，正好走pydantic-settings 的覆盖机制）
docker run -d --name video-audit -p 8000:8000 --env-file .env.qa video-audit:latest

# 运行 prod
docker run -d --name video-audit -p 8000:8000 --env-file .env.prod -e APP_ENV=prod video-audit:latest