# @Auther  : luoscorn
# @Time    : 2026/7/28: 17:01
# @File    : __init__.py.py
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# 通过环境变量 APP_ENV 选择环境(qa/prod) 默认qa
APP_ENV = os.getenv("APP_ENV", "qa")


class Settings(BaseSettings):
    """项目配置 优先级: 系统环境变量 > .env.{APP_ENV} 文件 > 类内默认值"""
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / f".env.{APP_ENV}",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # local mysql
    LOCAL_MYSQL_HOST: Optional[str] = None
    LOCAL_MYSQL_PORT: int = 3306
    LOCAL_MYSQL_USER: Optional[str] = None
    LOCAL_MYSQL_PASSWORD: Optional[str] = None
    LOCAL_MYSQL_DB: Optional[str] = None

    # 千问 API keys
    QWEN_API_KEY_1: Optional[str] = None
    QWEN_API_KEY_2: Optional[str] = None


settings = Settings()
