# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : rule_admin.py
"""打分规则管理页 — 服务端渲染 供人工维护各审核类型的评分项/文本规则

能力: 规则集列表 / 新建空白结构化草稿 / 派生草稿 / 表格编辑保存 / 预览提示词 / 发布版本
保护: HTTP Basic 口令(username 固定 admin，口令取 settings.ADMIN_PASSWORD)，未配置口令时整页禁用
规则改动只落在草稿上，发布后才对打分生效；历史版本保留为 archived，便于回滚与追溯
"""

import asyncio
import logging
import secrets
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from config import BASE_DIR, settings
from db.db import (
    count_rule_items,
    create_rule_draft,
    delete_rule_draft,
    get_active_rule_set,
    get_latest_draft,
    get_rule_items,
    get_rule_set,
    get_session,
    list_rule_sets,
    publish_rule_set,
    save_rule_draft,
)
from db.models import ScoreRuleSet
from services.qwen_client import format_rubric_preview
from utils.audit_type import AuditType
from utils.oss_client import get_object_url, upload_bytes
from utils.score_rules import items_from_rows

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/rules", tags=["规则管理"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 规则来源在库里是英文标记 页面直接展示英文对使用者没意义
_SOURCE_LABELS = {
    "web": "后台录入",
    "oss-xlsx": "Excel 导入",
    "oss-docx": "Word 导入",
    "local-txt": "内置文本",
}
templates.env.globals["SOURCE_LABELS"] = _SOURCE_LABELS

_basic = HTTPBasic(auto_error=False)
_ADMIN_USER = "admin"
_ITEM_COLUMNS = (
    "dimension",
    "number",
    "point",
    "evidence",
    "oral_visual_weight",
    "oral_keywords",
    "remark",
    "image_refs",
    "score",
)

# 操作图片上传限制: 只收图片小文件 直接进 OSS 后回填引用
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024


def require_admin(credentials: Optional[HTTPBasicCredentials] = Depends(_basic)) -> str:
    """Basic Auth 口令保护 未配置 ADMIN_PASSWORD 时禁用管理页"""
    if not settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="未配置 ADMIN_PASSWORD，规则管理页已禁用")
    valid = (
        credentials is not None
        and secrets.compare_digest(credentials.username, _ADMIN_USER)
        and secrets.compare_digest(credentials.password, settings.ADMIN_PASSWORD)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="认证失败",
            headers={"WWW-Authenticate": 'Basic realm="rule-admin"'},
        )
    return credentials.username


def _resolve_type(audit_type: str) -> AuditType:
    """路径参数转审核类型枚举 未知类型返回 404"""
    try:
        return AuditType(audit_type)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"未知审核类型: {audit_type}")


def _split_refs(text) -> list[str]:
    """操作图片引用拆行 每行一个 OSS key 或 http 直链"""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _image_url(ref: str) -> str:
    """图片引用转可展示地址: http 直链原样返回 其余按 OSS key 签名（桶未配置时返回空串）"""
    if ref.startswith(("http://", "https://")):
        return ref
    try:
        return get_object_url(oss_key=ref, expired=3600)
    except Exception:
        logger.warning("操作图片签名失败: %s", ref, exc_info=True)
        return ""


def _decorate_rows(rows: list[dict]) -> list[dict]:
    """给页面行补上可直接渲染的图片列表 不影响入库字段"""
    decorated = []
    for row in rows:
        refs = _split_refs(row.get("image_refs"))
        decorated.append({
            **row,
            "image_refs": "\n".join(refs),
            "images": [{"ref": ref, "url": _image_url(ref)} for ref in refs],
        })
    return decorated


def _rows_of(items) -> list[dict]:
    """数据库评分项转页面/入库行结构（保留备注、口述权重、关键词与图片引用）"""
    return [
        {
            "dimension": item.dimension or "",
            "number": item.number,
            "point": item.point,
            "evidence": item.evidence or "",
            "oral_visual_weight": item.oral_visual_weight or "",
            "oral_keywords": item.oral_keywords or "",
            "remark": item.remark or "",
            "image_refs": item.image_refs or "",
            "score": float(item.score),
        }
        for item in items
    ]


def _parse_full_score(raw) -> tuple[Optional[float], str]:
    """解析类目打分总分 允许带「分」字 留空表示不校验"""
    text = str(raw or "").strip().replace("分", "")
    if not text:
        return None, ""
    try:
        value = float(text)
    except ValueError:
        return None, f"类目打分总分「{text}」不是有效数字"
    if value <= 0:
        return None, "类目打分总分必须大于 0"
    return round(value, 2), ""


def _parse_item_rows(form) -> tuple[list[dict], str, list[dict]]:
    """
    解析评分项表单

    编号与评分点都为空的行视为未填写直接跳过；其余校验编号唯一、分值大于 0
    :return: (合法评分项, 错误信息, 原始行) — 原始行用于校验失败时回显用户已填内容
    """
    columns = {key: form.getlist(key) for key in _ITEM_COLUMNS}
    length = max((len(values) for values in columns.values()), default=0)

    rows: list[dict] = []
    raw_rows: list[dict] = []
    numbers: list[str] = []
    error = ""

    for idx in range(length):
        def cell(key: str, _idx: int = idx) -> str:
            values = columns[key]
            return str(values[_idx]).strip() if _idx < len(values) else ""

        number = cell("number")
        point = cell("point")
        if not number and not point:
            continue

        raw_rows.append({
            "dimension": cell("dimension"),
            "number": number,
            "point": point,
            "evidence": cell("evidence"),
            "oral_visual_weight": cell("oral_visual_weight"),
            "oral_keywords": cell("oral_keywords"),
            "remark": cell("remark"),
            "image_refs": cell("image_refs"),
            "score": cell("score"),
        })
        if error:
            continue

        if not number or not point:
            error = f"第 {len(raw_rows)} 行的编号与评分点都必须填写"
            continue
        if number in numbers:
            error = f"编号重复: {number}"
            continue
        try:
            score = float(cell("score"))
        except ValueError:
            error = f"「{number}」的分值不是有效数字"
            continue
        if score <= 0:
            error = f"「{number}」的分值必须大于 0"
            continue

        numbers.append(number)
        rows.append({
            "dimension": cell("dimension"),
            "number": number,
            "point": point,
            "evidence": cell("evidence"),
            "oral_visual_weight": cell("oral_visual_weight"),
            "oral_keywords": cell("oral_keywords"),
            "remark": cell("remark"),
            "image_refs": cell("image_refs"),
            "score": round(score, 2),
        })

    if error:
        return [], error, raw_rows
    if not rows:
        return [], "至少需要一个评分项", raw_rows
    return rows, "", raw_rows


async def _edit_context(
    session: AsyncSession,
    audit_type: AuditType,
    error: str = "",
    submitted: Optional[dict] = None,
    viewing_id: Optional[int] = None,
) -> dict:
    """构造编辑页上下文: 生效版、草稿、待编辑内容、历史版本

    :param viewing_id: 指定版本ID时额外返回该版本的只读内容（用于历史版本查看/预览）
    """
    active = await get_active_rule_set(session, audit_type.value)
    draft = await get_latest_draft(session, audit_type.value)
    target = draft or active

    if submitted is not None:
        kind = submitted.get("kind") or (target.kind if target else "rubric")
        rows = submitted.get("rows") or []
        content = submitted.get("content") or ""
        full_score = submitted.get("full_score")
        set_dimension = submitted.get("dimension") or ""
    else:
        kind = target.kind if target else "rubric"
        content = (target.content or "") if target is not None else ""
        rows = []
        if target is not None and target.kind == "rubric":
            rows = _rows_of(await get_rule_items(session, target.id))
        full_score = float(target.full_score) if target is not None and target.full_score is not None else None
        set_dimension = (target.dimension or "") if target is not None else ""

    rows = _decorate_rows(rows)

    # 各行维度去重: 存在多个维度时不在顶部预填 避免回写时把分组拍平成一个值
    dimensions: list[str] = []
    for row in rows:
        dimension = (row.get("dimension") or "").strip()
        if dimension and dimension not in dimensions:
            dimensions.append(dimension)
    if not set_dimension and len(dimensions) == 1:
        set_dimension = dimensions[0]

    total_score = None
    if kind == "rubric":
        total_score = 0.0
        for row in rows:
            try:
                total_score += float(row.get("score") or 0)
            except (TypeError, ValueError):
                # 回显非法分值时不参与合计 避免页面报错
                continue
        total_score = round(total_score, 2)

    check_pass = None
    if kind == "rubric" and full_score:
        check_pass = abs((total_score or 0) - full_score) < 0.005

    # 指定版本时把它单独取出来 不影响上方草稿的编辑上下文
    viewing = None
    if viewing_id:
        view_set = await get_rule_set(session, viewing_id)
        if view_set is not None and view_set.audit_type == audit_type.value:
            view_rows = []
            if view_set.kind == "rubric":
                view_rows = _decorate_rows(_rows_of(await get_rule_items(session, view_set.id)))
            # 带上表格抬头需要的字段 否则查看历史版时会显示成当前草稿的满分与来源
            view_full = float(view_set.full_score) if view_set.full_score is not None else None
            view_total = round(sum(float(row["score"]) for row in view_rows), 2) if view_rows else None
            viewing = {
                "rule_set": view_set,
                "rows": view_rows,
                "target": view_set,
                "full_score": view_full,
                "total_score": view_total,
                "source": view_set.source or "-",
                "check_pass": (
                    abs((view_total or 0) - view_full) < 0.005 if view_full else None
                ),
            }

    history = await list_rule_sets(session, audit_type.value)
    # 历史表格一并展示每个版本的规模 不用点进去就能比较
    history_counts = await count_rule_items(session, [rule_set.id for rule_set in history])

    return {
        "audit_type": audit_type.value,
        "active": active,
        "draft": draft,
        "target": target,
        "kind": kind,
        "rows": rows,
        "content": content,
        "total_score": total_score,
        "full_score": full_score,
        "check_pass": check_pass,
        "set_dimension": set_dimension,
        "dimensions": dimensions,
        "source": (target.source or "-") if target is not None else "-",
        "error": error,
        "preview": None,
        "history": history,
        "history_counts": history_counts,
        "viewing": viewing,
        "viewing_id": viewing_id,
    }


def _render_edit(request: Request, context: dict, operator: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="rule_admin/edit.html",
        context={**context, "operator": operator},
    )


async def _load_draft(
    session: AsyncSession, audit_type: AuditType, raw_id
) -> tuple[Optional[ScoreRuleSet], str]:
    """取表单提交的规则集 返回 (规则集, 错误提示) 不抛 400"""
    try:
        rule_set_id = int(raw_id or 0)
    except (TypeError, ValueError):
        rule_set_id = 0
    if not rule_set_id:
        return None, "表单里没有版本号，页面可能已过期，请刷新后重新操作"
    rule_set = await get_rule_set(session, rule_set_id)
    if rule_set is None or rule_set.audit_type != audit_type.value:
        return None, f"版本 #{rule_set_id} 不存在或不属于该审核类型，页面可能已过期，请刷新后重新操作"
    return rule_set, ""


_STATUS_LABELS = {"active": "生效中", "archived": "已归档", "draft": "草稿"}


async def _fail(
    session: AsyncSession,
    request: Request,
    audit_type: AuditType,
    operator: str,
    message: str,
    action: str = "规则操作失败",
) -> HTMLResponse:
    """失败统一回编辑页并说明原因 避免用户只看到一个 400 JSON"""
    logger.warning("%s: %s - %s", action, audit_type.value, message)
    context = await _edit_context(session, audit_type, error=message)
    return _render_edit(request, context, operator)


@router.get("", response_class=HTMLResponse, summary="规则集列表")
async def rule_list(
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """按审核类型列出当前生效版本与草稿状态"""
    sets = await list_rule_sets(session)
    state: dict[str, dict] = {}
    for rule_set in sets:
        entry = state.setdefault(rule_set.audit_type, {"active": None, "draft": None})
        if rule_set.status == "active" and entry["active"] is None:
            entry["active"] = rule_set
        elif rule_set.status == "draft" and entry["draft"] is None:
            entry["draft"] = rule_set

    rows = [
        {"audit_type": item.value, "state": state.get(item.value, {"active": None, "draft": None})}
        for item in AuditType
    ]
    # 一次查完所有版本的评分项数 避免逐行查询
    version_ids = [
        row["state"][key].id
        for row in rows
        for key in ("active", "draft")
        if row["state"][key] is not None
    ]
    counts = await count_rule_items(session, version_ids)
    for row in rows:
        for key in ("active", "draft"):
            rule_set = row["state"][key]
            if rule_set is not None:
                row["state"][f"{key}_items"] = counts.get(rule_set.id, 0)

    return templates.TemplateResponse(
        request=request,
        name="rule_admin/list.html",
        context={"rows": rows, "operator": operator},
    )


@router.get("/{audit_type}", response_class=HTMLResponse, summary="规则编辑页")
async def rule_edit(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
    version: Optional[int] = None,
):
    """展示待编辑内容: 优先草稿，没有草稿时展示生效版（只读，需先派生）

    传 version 时额外展开该历史版本的只读内容
    """
    context = await _edit_context(session, _resolve_type(audit_type), viewing_id=version)
    return _render_edit(request, context, operator)


@router.post("/{audit_type}/create", summary="新建空白结构化草稿")
async def rule_create(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """新建空白结构化草稿（只支持结构化评分项，不支持文本规则）"""
    target_type = _resolve_type(audit_type)
    if await get_latest_draft(session, target_type.value):
        return await _fail(
            session, request, target_type, operator,
            "该审核类型已有草稿，请先发布或删除", "规则新建失败",
        )

    draft = await create_rule_draft(
        session, target_type.value, "rubric", "web",
        remark="新建空白草稿", operator=operator,
    )
    logger.info("规则新建草稿: %s v%s", target_type.value, draft.version)
    return RedirectResponse(url=f"/admin/rules/{target_type.value}", status_code=303)


@router.post("/{audit_type}/derive", summary="从生效版派生草稿")
async def rule_derive(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """复制当前生效版本为新草稿，避免直接改动线上规则"""
    target_type = _resolve_type(audit_type)
    if await get_latest_draft(session, target_type.value):
        return await _fail(
            session, request, target_type, operator,
            "该审核类型已有草稿，请先发布或删除", "规则派生失败",
        )
    active = await get_active_rule_set(session, target_type.value)
    if active is None:
        return await _fail(
            session, request, target_type, operator,
            "该审核类型还没有生效版本，请先「新建空白结构化草稿」逐项录入并发布", "规则派生失败",
        )
    if active.kind != "rubric":
        return await _fail(
            session, request, target_type, operator,
            "当前生效版是文本规则，不支持派生成结构化草稿；请用「新建空白结构化草稿」逐项录入",
            "规则派生失败",
        )

    draft = await create_rule_draft(
        session, target_type.value, "rubric", active.source,
        remark=f"派生自 v{active.version}", operator=operator,
    )
    # 派生时带上规则集级的维度与类目打分总分 否则新草稿会丢掉校验基准
    draft.dimension = active.dimension
    draft.full_score = active.full_score
    rows = _rows_of(await get_rule_items(session, active.id))
    await save_rule_draft(session, draft, rows, operator=operator)
    return RedirectResponse(url=f"/admin/rules/{target_type.value}", status_code=303)


@router.post("/{audit_type}/save", response_class=HTMLResponse, summary="保存草稿")
async def rule_save(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """保存草稿内容（全量覆盖），校验失败时回显已填内容"""
    target_type = _resolve_type(audit_type)
    form = await request.form()
    rule_set, load_error = await _load_draft(session, target_type, form.get("rule_set_id"))
    if load_error:
        return await _fail(session, request, target_type, operator, load_error, "规则保存失败")
    if rule_set.status != "draft":
        label = _STATUS_LABELS.get(rule_set.status, rule_set.status)
        return await _fail(
            session, request, target_type, operator,
            f"v{rule_set.version} 当前是{label}状态，不能直接编辑；如需修改请先「从生效版派生草稿」",
            "规则保存失败",
        )

    remark = str(form.get("change_note") or "").strip()

    if rule_set.kind == "text":
        content = str(form.get("content") or "")
        if not content.strip():
            context = await _edit_context(
                session, target_type, error="规则内容不能为空",
                submitted={"kind": "text", "rows": [], "content": content},
            )
            return _render_edit(request, context, operator)
        await save_rule_draft(session, rule_set, [], content=content, remark=remark, operator=operator)
    else:
        set_dimension = str(form.get("set_dimension") or "").strip()
        full_score, score_error = _parse_full_score(form.get("full_score"))
        rows, error, raw_rows = _parse_item_rows(form)
        error = error or score_error
        if error:
            context = await _edit_context(
                session, target_type, error=error,
                submitted={
                    "kind": "rubric", "rows": raw_rows, "content": "",
                    "full_score": full_score, "dimension": set_dimension,
                },
            )
            return _render_edit(request, context, operator)
        for row in rows:
            if set_dimension:
                # 顶部统一维度优先: 填了就覆盖每行原有维度 留空则保留各行分组
                row["dimension"] = set_dimension
            # 图片引用统一成一行一个 避免入库空行
            row["image_refs"] = "\n".join(_split_refs(row.get("image_refs")))
        rule_set.dimension = set_dimension or None
        rule_set.full_score = full_score
        await save_rule_draft(session, rule_set, rows, remark=remark, operator=operator)

    logger.info("规则草稿保存: %s v%s 类型=%s", target_type.value, rule_set.version, rule_set.kind)
    return RedirectResponse(url=f"/admin/rules/{target_type.value}", status_code=303)


@router.post("/{audit_type}/publish", summary="发布草稿")
async def rule_publish(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """发布草稿: 同类型旧生效版自动归档，新版本立即对打分生效"""
    target_type = _resolve_type(audit_type)
    form = await request.form()
    rule_set, load_error = await _load_draft(session, target_type, form.get("rule_set_id"))
    if load_error:
        return await _fail(session, request, target_type, operator, load_error, "规则发布失败")
    if rule_set.status != "draft":
        label = _STATUS_LABELS.get(rule_set.status, rule_set.status)
        return await _fail(
            session, request, target_type, operator,
            f"v{rule_set.version} 已经是{label}状态，无需重复发布；要再改请从生效版派生新草稿",
            "规则发布失败",
        )

    if rule_set.kind == "rubric":
        if not await get_rule_items(session, rule_set.id):
            return await _fail(
                session, request, target_type, operator,
                "草稿里还没有任何评分项，无法发布", "规则发布失败",
            )
    elif not (rule_set.content or "").strip():
        return await _fail(
            session, request, target_type, operator,
            "草稿规则内容为空，无法发布", "规则发布失败",
        )

    await publish_rule_set(session, rule_set)
    logger.info("规则发布: %s v%s (by %s)", target_type.value, rule_set.version, operator)
    return RedirectResponse(url=f"/admin/rules/{target_type.value}", status_code=303)


@router.post("/{audit_type}/delete", summary="删除草稿")
async def rule_delete(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """删除草稿（生效版与历史版本不允许删除）"""
    target_type = _resolve_type(audit_type)
    form = await request.form()
    rule_set, load_error = await _load_draft(session, target_type, form.get("rule_set_id"))
    if load_error:
        return await _fail(session, request, target_type, operator, load_error, "规则删除失败")
    if rule_set.status != "draft":
        label = _STATUS_LABELS.get(rule_set.status, rule_set.status)
        return await _fail(
            session, request, target_type, operator,
            f"只能删除草稿；v{rule_set.version} 当前是{label}状态，不能删除",
            "规则删除失败",
        )
    logger.info("规则草稿删除: %s v%s", target_type.value, rule_set.version)
    await delete_rule_draft(session, rule_set)
    return RedirectResponse(url=f"/admin/rules/{target_type.value}", status_code=303)


@router.post("/{audit_type}/upload-image", summary="上传评分项操作图片")
async def rule_upload_image(
    audit_type: str,
    file: UploadFile = File(...),
    operator: str = Depends(require_admin),
):
    """评分项操作图片上传到 OSS 返回对象键与临时可访问地址 供页面回填图片列"""
    target_type = _resolve_type(audit_type)
    ext = _IMAGE_TYPES.get((file.content_type or "").lower())
    if not ext:
        raise HTTPException(status_code=400, detail="只支持 jpg / png / webp / gif 图片")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 5MB")

    oss_key = f"rule-admin/{target_type.value}/{uuid.uuid4().hex}{ext}"
    try:
        await asyncio.to_thread(upload_bytes, data, oss_key)
        url = await asyncio.to_thread(get_object_url, oss_key, None, 3600)
    except Exception:
        logger.exception("操作图片上传失败: %s", target_type.value)
        raise HTTPException(status_code=502, detail="图片上传到 OSS 失败，请检查 OSS 配置")

    logger.info("操作图片上传: %s %s (by %s)", target_type.value, oss_key, operator)
    return JSONResponse({"key": oss_key, "url": url})


@router.get("/{audit_type}/preview", response_class=HTMLResponse, summary="预览提示词")
async def rule_preview(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
    version: Optional[int] = None,
):
    """按待编辑内容渲染最终写入提示词的规则文本

    传 version 时渲染该历史版本 供回滚前对照
    """
    target_type = _resolve_type(audit_type)
    context = await _edit_context(session, target_type, viewing_id=version)
    if version is not None:
        viewing = context.get("viewing")
        if viewing is None:
            context["error"] = f"版本 v{version} 不存在或不属于该审核类型"
            return _render_edit(request, context, operator)
        rule_set = viewing["rule_set"]
        context["preview"] = (
            format_rubric_preview(items_from_rows(viewing["rows"]))
            if rule_set.kind == "rubric"
            else (rule_set.content or "")
        )
        context["preview_title"] = f"v{rule_set.version} · {_STATUS_LABELS.get(rule_set.status, rule_set.status)}"
    elif context["kind"] == "rubric":
        context["preview"] = format_rubric_preview(items_from_rows(context["rows"]))
        context["preview_title"] = "当前待编辑内容"
    else:
        context["preview"] = context["content"]
        context["preview_title"] = "当前待编辑内容"
    return _render_edit(request, context, operator)
