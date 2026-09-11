# @Auther  : luoscorn
# @Time    : 2026/9/11
# @File    : oss_client.py
"""阿里云 OSS 工具模块"""

import os
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import oss2
import requests

from config import BASE_DIR, settings

# 进度回调类型: (已上传字节数, 总字节数)
ProgressCallback = Callable[[int, int], None]


def get_oss_bucket(bucket: Optional[str] = None) -> oss2.Bucket:
    """获取 OSS Bucket 客户端实例"""
    auth = oss2.Auth(settings.ALIYUN_ACCESS_KEY_ID, settings.ALIYUN_ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, settings.OSS_ENDPOINT, bucket or settings.OSS_BUCKET)


def upload_file(
    local_path: str,
    oss_key: str,
    bucket: Optional[str] = None,
    part_size: int = 10,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    上传本地文件到 OSS 大文件自动断点续传 支持进度回调

    :param local_path: 本地文件路径
    :param oss_key: OSS 上的对象键(路径)
    :param bucket: 存储桶名称，默认使用配置中的 OSS_BUCKET
    :param part_size: 分块大小(MB)，默认 10MB
    :param progress_callback: 进度回调函数 (已上传字节数, 总字节数)
    :return: 文件的 OSS 访问 URL
    """
    bucket_obj = get_oss_bucket(bucket)
    part_size_bytes = part_size * 1024 * 1024

    def _progress(bytes_consumed: int, total_bytes: Optional[int]):
        if progress_callback and total_bytes:
            progress_callback(bytes_consumed, total_bytes)

    oss2.resumable_upload(
        bucket_obj,
        oss_key,
        local_path,
        multipart_threshold=part_size_bytes,
        part_size=part_size_bytes,
        progress_callback=_progress,
    )

    host = settings.OSS_ENDPOINT.replace("https://", "").replace("http://", "")
    url = f"https://{bucket_obj.bucket_name}.{host}/{oss_key}"
    print(f"上传成功: {local_path} -> {url}")
    return url


def list_objects(
    prefix: str = "",
    bucket: Optional[str] = None,
    max_keys: int = 100,
) -> list[str]:
    """
    列出 OSS 存储桶中指定前缀下的所有对象 Key

    :param prefix: 对象键前缀，如 "video/上颌印模制取/"
    :param bucket: 存储桶名称，默认使用配置中的 OSS_BUCKET
    :param max_keys: 最多返回条数，默认 100
    :return: 对象 Key 列表
    """
    bucket_obj = get_oss_bucket(bucket)
    keys = []
    for obj in oss2.ObjectIterator(bucket_obj, prefix=prefix, max_keys=1000):
        keys.append(obj.key)
        if len(keys) >= max_keys:
            break
    return keys


def get_object_url(
    oss_key: str,
    bucket: Optional[str] = None,
    expired: int = 3600,
) -> str:
    """
    获取 OSS 对象的预签名下载 URL

    :param oss_key: OSS 上的对象键(路径)
    :param bucket: 存储桶名称，默认使用配置中的 OSS_BUCKET
    :param expired: 签名有效期(秒)，默认 1 小时
    :return: 预签名下载 URL
    """
    bucket_obj = get_oss_bucket(bucket)
    # slash_safe=True: key 含中文与斜杠 必须开启 否则签名与实际URL不一致导致403
    return bucket_obj.sign_url("GET", oss_key, expired, slash_safe=True)


def object_stat(oss_key: str, bucket: Optional[str] = None) -> Optional[int]:
    """
    查询 OSS 对象大小

    :param oss_key: OSS 上的对象键(路径)
    :return: 对象字节数，对象不存在返回 None
    """
    bucket_obj = get_oss_bucket(bucket)
    try:
        return bucket_obj.head_object(oss_key).content_length
    except oss2.exceptions.OssError as e:
        if e.status == 404:
            return None
        raise


def download_file(
    url: str,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    从 OSS 预签名 URL 下载文件到本地 doc/ 目录

    :param url: OSS 预签名下载链接
    :param save_dir: 保存目录，默认为项目根目录下的 doc/
    :param filename: 指定保存文件名，默认从 URL 路径中解析
    :param progress_callback: 进度回调函数 (已下载字节数, 总字节数)
    :return: 本地保存的文件路径
    """
    save_dir = save_dir or str(BASE_DIR / "doc")
    os.makedirs(save_dir, exist_ok=True)

    # 从 URL 路径中解析文件名（去掉查询参数，URL 解码）
    if not filename:
        parsed = urlparse(url)
        filename = os.path.basename(unquote(parsed.path))

    save_path = os.path.join(save_dir, filename)

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total_size:
                progress_callback(downloaded, total_size)

    print(f"下载完成: {save_path} ({downloaded / 1024 / 1024:.2f}MB)")
    return save_path
