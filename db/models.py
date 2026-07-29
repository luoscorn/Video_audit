# @Auther  : luoscorn
# @Time    : 2026/7/29: 17:00
# @File    : models.py
from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, Numeric, String, func, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VideoAuditTask(Base):
    __tablename__ = "video_audit_task"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_status = Column(Integer, default=0, nullable=False, comment="任务状态：0-已接收 1-进行中 2-已完成 3-异常")
    audit_type = Column(String(255), nullable=False, comment="审核类型")
    oss_url = Column(String(255), nullable=False, comment="oss地址")
    create_time = Column(DateTime, default=func.now(), comment="任务创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="任务更新时间")


class VideoAuditTaskResult(Base):
    __tablename__ = "video_audit_task_result"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("video_audit_task.id"), nullable=False, comment="关联 video_audit_task.id")
    score = Column(Numeric(5, 2), comment="审核分数")
    result_json = Column(JSON, comment="审核结果原始JSON")
    create_time = Column(DateTime, default=func.now(), comment="结果创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="结果更新时间")

    __table_args__ = (
        Index("idx_task_id", "task_id"),
    )
