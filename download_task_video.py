# @Auther  : luoscorn
# @Time    : 2026/11/9
# @File    : download_task_video.py
"""按任务号把 OSS 上的原始视频下载到本地 供人工核对评分

用法:
    python download_task_video.py 126
    python download_task_video.py 126 --out ~/Desktop --name 复核样片.mp4
"""

import argparse
import asyncio
import os

import requests

from db.db import AsyncSessionLocal, get_task
from services.audit_worker import _extract_object_key
from utils.oss_client import get_object_url

_CHUNK = 1024 * 256


async def _fetch_task(task_id: int) -> tuple[str, str]:
    """取任务的审核类型与原始 OSS 路径"""
    async with AsyncSessionLocal() as session:
        task = await get_task(session, task_id)
        if task is None:
            raise SystemExit(f"任务 {task_id} 不存在")
        if not task.oss_url:
            raise SystemExit(f"任务 {task_id} 的 oss_url 为空 无法下载")
        return task.audit_type, task.oss_url


def _download(oss_key: str, dest: str) -> int:
    """下载到本地 返回字节数"""
    url = get_object_url(oss_key, expired=3600)
    out_dir = os.path.dirname(dest)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    total = 0
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fp:
            for chunk in resp.iter_content(_CHUNK):
                if not chunk:
                    continue
                fp.write(chunk)
                total += len(chunk)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="下载指定任务的原始视频")
    parser.add_argument("task_id", type=int, help="任务 id")
    parser.add_argument("--out", default="~/Downloads", help="输出目录 默认 ~/Downloads")
    parser.add_argument("--name", default="", help="文件名 默认 {审核类型}_{任务id}.mp4")
    args = parser.parse_args()

    try:
        audit_type, oss_url = asyncio.run(_fetch_task(args.task_id))
    except Exception as exc:
        # 连不上库时不要把 SQLAlchemy 那一大段堆栈丢给使用者
        raise SystemExit(f"查询任务失败: {exc}") from None
    oss_key = _extract_object_key(oss_url)
    suffix = os.path.splitext(oss_key)[1] or ".mp4"
    dest = os.path.join(
        os.path.expanduser(args.out), args.name or f"{audit_type}_{args.task_id}{suffix}"
    )
    size = _download(oss_key, dest)
    print(f"{dest}  {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
