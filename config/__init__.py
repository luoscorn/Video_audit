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
    # 复核片段在窗口内重新抽帧的张数与长边上限 密度与分辨率都高于首轮
    # 手指抵基座这类接触点在 1024 宽的帧里只剩十几个像素 复核得回到原片分辨率才看得清
    MEDIA_SUPPLEMENT_FRAMES: int = 12
    MEDIA_SUPPLEMENT_FRAME_WIDTH: int = 1920
    # 规范参考图: 首轮是否携带
    # 参考图已声明「只帮助理解不构成要求」 却占掉首轮三分之一的图片预算
    # 实测首轮把 9 项扫完只花 23 秒 属于带着参考图在扫读 默认不送
    MEDIA_REFERENCE_IMAGES_ENABLED: bool = False
    # 扣分项投票: 只对已扣分的评分项用同一份证据重判 逐项取中位数
    # 同一帧首轮判「无名指悬空」、另一轮判「抵在模型下颌基座」 单次结论不可信
    # 满分项不花调用 全对时零成本; 轮数为 4 时共五票 中位数才拉得开两极
    MEDIA_VOTE_ENABLED: bool = True
    MEDIA_VOTE_ROUNDS: int = 4
    # 冲突仲裁: 各项判定对同一事实给出相反结论时拿同一份证据重判这几项
    # 补证是链路最后一环 它改完分数后没人回头检查其余项 矛盾结论会直接入库
    # 检测只走纯文本 无矛盾时不花带证据的调用
    MEDIA_CONFLICT_ENABLED: bool = True
    # 千问思考模式: 把推理挪出 reason 字段 但更慢更贵 效果未经 A/B 确认前默认关闭
    QWEN_ENABLE_THINKING: bool = False
    QWEN_THINKING_BUDGET: int = 4096
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
