# @Auther  : luoscorn
# @Time    : 2026/7/29: 17:10
# @File    : input.py
from pydantic import BaseModel, Field

from utils.audit_type import AuditType

# 任务状态码 -> 描述
TASK_STATUS_DESC = {0: "已接收", 1: "进行中", 2: "已完成", 3: "异常"}


class CreateTaskReq(BaseModel):
    """创建审核任务入参"""
    audit_type: AuditType = Field(..., description="审核类型 枚举值见 AuditType")
    oss_url: str = Field(..., max_length=255, description="OSS路径 格式: {bucket}/{oss_key} 如 wyxtapp/video/xxx/yyy.mp4")


class PageQuery(BaseModel):
    """分页查询参数"""
    page: int = Field(1, ge=1, description="页码 从1开始")
    page_size: int = Field(10, ge=1, le=100, description="每页条数 默认10 最大100")
