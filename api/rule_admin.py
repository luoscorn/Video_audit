# @Auther  : luoscorn
# @Time    : 2026/11/4
# @File    : rule_admin.py
"""打分规则管理页 — 服务端渲染 供人工维护各审核类型的评分项/文本规则

能力: 规则集列表 / 新建空白结构化草稿 / 派生草稿 / 表格编辑保存 / 预览提示词 / 发布版本
保护: HTTP Basic 口令(username 固定 admin，口令取 settings.ADMIN_PASSWORD)，未配置口令时整页禁用
规则改动只落在草稿上，发布后才对打分生效；历史版本保留为 archived，便于回滚与追溯
"""

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from config import BASE_DIR, settings
from db.db import (
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
from utils.score_rules import items_from_rows

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/rules", tags=["规则管理"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_basic = HTTPBasic(auto_error=False)
_ADMIN_USER = "admin"
_ITEM_COLUMNS = ("dimension", "number", "point", "evidence", "remark", "score")


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


def _rows_of(items) -> list[dict]:
    """数据库评分项转页面行结构（保留备注）"""
    return [
        {
            "dimension": item.dimension or "",
            "number": item.number,
            "point": item.point,
            "evidence": item.evidence or "",
            "remark": item.remark or "",
            "score": float(item.score),
        }
        for item in items
    ]


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
            "remark": cell("remark"),
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
            "remark": cell("remark"),
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
) -> dict:
    """构造编辑页上下文: 生效版、草稿、待编辑内容、历史版本"""
    active = await get_active_rule_set(session, audit_type.value)
    draft = await get_latest_draft(session, audit_type.value)
    target = draft or active

    if submitted is not None:
        kind = submitted.get("kind") or (target.kind if target else "rubric")
        rows = submitted.get("rows") or []
        content = submitted.get("content") or ""
    else:
        kind = target.kind if target else "rubric"
        content = (target.content or "") if target is not None else ""
        rows = []
        if target is not None and target.kind == "rubric":
            rows = _rows_of(await get_rule_items(session, target.id))

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

    return {
        "audit_type": audit_type.value,
        "active": active,
        "draft": draft,
        "target": target,
        "kind": kind,
        "rows": rows,
        "content": content,
        "total_score": total_score,
        "error": error,
        "preview": None,
        "history": await list_rule_sets(session, audit_type.value),
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
):
    """展示待编辑内容: 优先草稿，没有草稿时展示生效版（只读，需先派生）"""
    context = await _edit_context(session, _resolve_type(audit_type))
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
        rows, error, raw_rows = _parse_item_rows(form)
        if error:
            context = await _edit_context(
                session, target_type, error=error,
                submitted={"kind": "rubric", "rows": raw_rows, "content": ""},
            )
            return _render_edit(request, context, operator)
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


@router.get("/{audit_type}/preview", response_class=HTMLResponse, summary="预览提示词")
async def rule_preview(
    audit_type: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    operator: str = Depends(require_admin),
):
    """按待编辑内容渲染最终写入提示词的规则文本"""
    target_type = _resolve_type(audit_type)
    context = await _edit_context(session, target_type)
    if context["kind"] == "rubric":
        context["preview"] = format_rubric_preview(items_from_rows(context["rows"]))
    else:
        context["preview"] = context["content"]
    return _render_edit(request, context, operator)
