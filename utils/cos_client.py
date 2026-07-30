# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : cos_client.py
"""腾讯云 COS 工具模块"""

import os
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import requests
from qcloud_cos import CosConfig, CosS3Client

from config import settings, BASE_DIR

# 进度回调类型: (已上传字节数, 总字节数)
ProgressCallback = Callable[[int, int], None]


def get_cos_client() -> CosS3Client:
    """获取 COS 客户端实例"""
    config = CosConfig(
        Region=settings.COS_REGION,
        SecretId=settings.TENCENT_SECRET_ID,
        SecretKey=settings.TENCENT_SECRET_KEY,
    )
    return CosS3Client(config)


def upload_file(
    local_path: str,
    cos_key: str,
    bucket: Optional[str] = None,
    part_size: int = 10,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    上传本地文件到 COS，支持进度回调

    :param local_path: 本地文件路径
    :param cos_key: COS 上的对象键(路径)
    :param bucket: 存储桶名称，默认使用配置中的 COS_BUCKET
    :param part_size: 分块大小(MB)，默认 10MB
    :param progress_callback: 进度回调函数 (已上传字节数, 总字节数)
    :return: 文件的 COS 访问 URL
    """
    client = get_cos_client()
    bucket = bucket or settings.COS_BUCKET
    part_size_bytes = part_size * 1024 * 1024
    file_size = os.path.getsize(local_path)

    if file_size <= part_size_bytes:
        # 小文件直接上传
        with open(local_path, "rb") as fp:
            client.put_object(Bucket=bucket, Key=cos_key, Body=fp)
        if progress_callback:
            progress_callback(file_size, file_size)
    else:
        # 大文件手动分块上传
        _multipart_upload(client, bucket, cos_key, local_path, file_size, part_size_bytes, progress_callback)

    url = f"https://{bucket}.cos.{settings.COS_REGION}.myqcloud.com/{cos_key}"
    print(f"上传成功: {local_path} -> {url}")
    return url


def _multipart_upload(
    client: CosS3Client,
    bucket: str,
    cos_key: str,
    local_path: str,
    file_size: int,
    part_size_bytes: int,
    progress_callback: Optional[ProgressCallback],
) -> None:
    """手动分块上传，支持进度回调"""
    # 1. 初始化分块上传
    resp = client.create_multipart_upload(Bucket=bucket, Key=cos_key)
    upload_id = resp["UploadId"]

    parts = []
    uploaded = 0
    part_number = 1

    try:
        with open(local_path, "rb") as fp:
            while True:
                data = fp.read(part_size_bytes)
                if not data:
                    break

                resp = client.upload_part(
                    Bucket=bucket,
                    Key=cos_key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=data,
                )

                parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
                uploaded += len(data)
                part_number += 1

                if progress_callback:
                    progress_callback(uploaded, file_size)

        # 2. 完成分块上传
        client.complete_multipart_upload(
            Bucket=bucket,
            Key=cos_key,
            UploadId=upload_id,
            MultipartUpload={"Part": parts},
        )
    except Exception:
        # 上传失败时取消
        client.abort_multipart_upload(Bucket=bucket, Key=cos_key, UploadId=upload_id)
        raise


def list_objects(
    prefix: str = "",
    bucket: Optional[str] = None,
    max_keys: int = 100,
) -> list[str]:
    """
    列出 COS 存储桶中指定前缀下的所有对象 Key

    :param prefix: 对象键前缀，如 "video/上颌印模制取/"
    :param bucket: 存储桶名称，默认使用配置中的 COS_BUCKET
    :param max_keys: 最多返回条数，默认 100
    :return: 对象 Key 列表
    """
    client = get_cos_client()
    bucket = bucket or settings.COS_BUCKET
    keys = []
    marker = ""

    while len(keys) < max_keys:
        resp = client.list_objects(
            Bucket=bucket,
            Prefix=prefix,
            Marker=marker,
            MaxKeys=min(max_keys - len(keys), 1000),
        )
        contents = resp.get("Contents", [])
        if not contents:
            break
        keys.extend(obj["Key"] for obj in contents)
        if resp.get("IsTruncated") == "false":
            break
        marker = resp.get("NextMarker", "")

    return keys


def get_object_url(
    cos_key: str,
    bucket: Optional[str] = None,
    expired: int = 3600,
) -> str:
    """
    获取 COS 对象的预签名下载 URL

    :param cos_key: COS 上的对象键(路径)
    :param bucket: 存储桶名称，默认使用配置中的 COS_BUCKET
    :param expired: 签名有效期(秒)，默认 1 小时
    :return: 预签名下载 URL
    """
    client = get_cos_client()
    bucket = bucket or settings.COS_BUCKET
    url = client.get_presigned_url(
        Method="GET",
        Bucket=bucket,
        Key=cos_key,
        Expired=expired,
    )
    return url


def download_file(
    url: str,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    从 COS 预签名 URL 下载文件到本地 doc/ 目录

    :param url: COS 预签名下载链接
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


if __name__ == "__main__":
    import sys

    def print_progress(current: int, total: int):
        pct = current / total * 100
        print(f"\r进度: {pct:.1f}% ({current / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB)", end="")

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        # 下载示例: python cos_client.py download <url>
        url = sys.argv[2] if len(sys.argv) > 2 else (
            "https://wiya-app-1346197003.cos.ap-chengdu.myqcloud.com/video/"
            "%E4%B8%8A%E9%A2%8C%E5%8D%B0%E6%A8%A1%E5%88%B6%E5%8F%96/"
            "%E4%B8%8A%E4%B8%8B%E7%89%99%E5%88%97%E5%8D%B0%E6%A8%A1%E5%88%B6%E5%8F%96.docx"
            "?q-sign-algorithm=sha1"
            "&q-ak=AKIDyuNx1TbsVRc1ijyqfNPhyLrsTfkESWPj4pnNEMQO3uVD30BcRLBUfBSa2RInCMEF"
            "&q-sign-time=1785393741;1785397341"
            "&q-key-time=1785393741;1785397341"
            "&q-header-list=host"
            "&q-url-param-list="
            "&q-signature=ecbda8ab4c81793b028556b42171b746403d6a90"
            "&x-cos-security-token=nkuN4FkyW4WnnKvl14tVXDvgOsZx52Fa53d70963f291358332723f6943fe9f60GOHyEnYZMnFnBh0__r-B_JXBhJCl-LQPrBzoGzUVvTeETRUrMBlLGYYMos186W0XeHzsxqnRCp9B6VagGZ68QT8aSAm9b9W3-6eMKFPVpYNZYHx9nuo95daV3-vD-qdCV0aPspVPdrDdqB_8YdVS4dQF7ef9U5CspA3jb61i169i9qLwkW7uL7LydBsrhmr3mYZmtTWeESHXAx1fXGfI1ehks6Rsk-4FWoi2kPqrumgUBe_1_62fslj2U9OrjgR9l3keq4bFKm7dB1wztSnwGg"
        )
        path = download_file(url=url, progress_callback=print_progress)
        print(f"\n文件已保存: {path}")
    else:
        # 上传示例: python cos_client.py
        base_dir = Path(__file__).resolve().parent.parent
        video_path = base_dir / "doc" / "上牙槽后神经阻滞麻醉标准视频.mp4"

        if not video_path.exists():
            print(f"文件不存在: {video_path}")
        else:
            url = upload_file(
                local_path=str(video_path),
                cos_key="video/上牙槽后神经阻滞麻醉标准视频.mp4",
                progress_callback=print_progress,
            )
            print(f"\n访问地址: {url}")
