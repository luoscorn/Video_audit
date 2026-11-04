# @Auther  : luoscorn
# @Time    : 2026/7/29: 17:00
# @File    : models.py
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
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


class ScoreRuleSet(Base):
    """打分规则集: 同一审核类型多版本共存 只有一版生效"""
    __tablename__ = "score_rule_set"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_type = Column(String(255), nullable=False, comment="审核类型 对应 AuditType 的值")
    version = Column(Integer, nullable=False, comment="同一审核类型的版本号 从1递增")
    kind = Column(String(16), nullable=False, default="rubric", comment="rubric-结构化评分项 text-整段文本规则")
    status = Column(String(16), nullable=False, default="draft", comment="draft-草稿 active-生效 archived-归档")
    source = Column(String(32), default="web", comment="来源: oss-xlsx/oss-docx/local-txt/web")
    content = Column(Text, comment="kind=text 时的规则全文")
    remark = Column(String(255), comment="变更说明")
    operator = Column(String(64), comment="最后修改人")
    create_time = Column(DateTime, default=func.now(), comment="创建时间")
    update_time = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    published_at = Column(DateTime, comment="发布时间")

    __table_args__ = (
        UniqueConstraint("audit_type", "version", name="uk_rule_audit_version"),
        Index("idx_rule_type_status", "audit_type", "status"),
    )


class ScoreRuleItem(Base):
    """结构化评分项: 隶属于某个规则集版本"""
    __tablename__ = "score_rule_item"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_set_id = Column(Integer, ForeignKey("score_rule_set.id"), nullable=False, comment="关联 score_rule_set.id")
    dimension = Column(String(64), comment="一级维度")
    number = Column(String(16), nullable=False, comment="评分项编号")
    point = Column(String(255), nullable=False, comment="细分评分点")
    evidence = Column(String(500), comment="视频识别依据")
    remark = Column(String(255), comment="备注 写入打分提示词作为该项补充判定说明")
    score = Column(Numeric(6, 2), nullable=False, comment="分值")
    sort_order = Column(Integer, default=0, comment="展示与提示词中的顺序")

    __table_args__ = (
        Index("idx_rule_set_id", "rule_set_id"),
    )
