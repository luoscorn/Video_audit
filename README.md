# Video_audit


视频审计


# 构建
docker build -t video-audit:latest .

# 运行 qa（配置通过 --env-file 注入，环境变量优先级高于 .env 文件，正好走pydantic-settings 的覆盖机制）
docker run -d --name video-audit -p 8022:8022 --env-file .env.qa video-audit:latest

docker run -d --name video-audit --network host --env-file .env.qa video-audit:latest

# 运行 prod
docker run -d --name video-audit -p 8022:8022 --env-file .env.prod -e APP_ENV=prod video-audit:latest

接口文档：
[http://127.0.0.1:8022/docs](http://127.0.0.1:8022/docs)
[http://127.0.0.1:8022/redoc](http://127.0.0.1:8022/redoc) 

