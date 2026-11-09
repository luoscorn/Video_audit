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

    # 千问模型 便于供应商下线模型时改配置而不改代码
    QWEN_VL_MODEL: str = "qwen3-vl-plus"
    QWEN_ASR_MODEL: str = "qwen3-asr-flash"

    # 视频切片抽帧 + 语音转写开关 关闭后回退为整段视频打分
    AUDIT_MEDIA_ENABLED: bool = True
    # 关键帧抽取: 总帧数上限与长边像素上限 直接影响画面识别粒度与 token 成本
    MEDIA_MAX_FRAMES: int = 24
    MEDIA_FRAME_WIDTH: int = 1024
    # 二次补证: 仅对证据受限的评分项切连续片段重判 关闭后只用一轮关键帧
    MEDIA_SUPPLEMENT_ENABLED: bool = True
    MEDIA_SUPPLEMENT_CLIPS: int = 6
    # 外部 ffmpeg 可执行文件路径 为空时依次尝试 PATH 与 imageio-ffmpeg 内置二进制
    FFMPEG_BINARY: Optional[str] = None
    # 规则管理页口令 为空则管理页禁用(用户名固定 admin)
    ADMIN_PASSWORD: Optional[str] = None

    # 阿里云 OSS
    ALIYUN_ACCESS_KEY_ID: Optional[str] = None
    ALIYUN_ACCESS_KEY_SECRET: Optional[str] = None
    OSS_BUCKET: str = "wyxtapp"
    OSS_ENDPOINT: str = "https://oss-cn-chengdu.aliyuncs.com"


settings = Settings()
