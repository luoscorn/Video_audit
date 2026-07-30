# @Auther  : luoscorn
# @Time    : 2026/7/29: 16:51
# @File    : audit_type.py
from enum import Enum
from pathlib import Path

from config import BASE_DIR


class AuditType(str, Enum):
    """审核类型枚举 值与 cos 桶 video/ 下的目录名一一对应"""
    PSA_NERVE_BLOCK = "上牙槽后神经阻滞麻醉"
    MAXILLARY_IMPRESSION = "上颌印模制取"
    IA_NERVE_BLOCK = "下牙槽神经阻滞麻醉"
    SUBMANDIBULAR_GLAND_EXAM = "下颌下腺检查"
    INTRAORAL_SUTURE = "口内缝合"
    PERCUSSION_EXAM = "叩诊检查"
    POSTERIOR_CROWN_PREP = "后牙铸造全冠牙体预备"
    OXYGEN_THERAPY = "吸氧术"
    OCCLUSION_EXAM = "咬合关系检查"
    CPR = "心肺复苏"
    PALPATION_EXAM = "扪诊检查"
    PROBING_EXAM = "探诊检查"
    MODIFIED_BASS_BRUSHING = "改良BASS刷牙法"
    MODIFIED_CPI_PROBING = "改良CPI探诊检查"
    MOBILITY_EXAM = "松动度检查"
    RUBBER_DAM_ISOLATION = "橡皮障隔离术"
    PERIODONTAL_PROBING = "牙周探诊"
    TMJ_EXAM = "颞下颌关节检查"
    TOOTH_EXTRACTION = "牙拔除术"
    FLOSS_GUIDANCE = "牙线使用指导"
    PULP_VITALITY_TEST = "牙髓活力测试"
    MOLAR_ACCESS_OPENING = "磨牙开髓术"
    EXTRACTED_MOLAR_CAVITY_PREP = "离体磨牙复面洞制备术"
    PIT_FISSURE_SEALING = "窝沟封闭"
    ABSCESS_INCISION_DRAINAGE = "脓肿切开引流术"
    MUCOSA_DISINFECTION = "黏膜消毒"
    SUPRAGINGIVAL_SCALING = "龈上洁治"
    BLOOD_PRESSURE_MEASUREMENT = "血压测量"
    VISUAL_EXAM = "视诊检查"


# AuditType → COS 上对应的 docx 文件 key（video/<目录>/<文件名>.docx）
AUDIT_SCORE_DOC: dict[AuditType, str] = {
    AuditType.PSA_NERVE_BLOCK:             "video/上牙槽后神经阻滞麻醉/上牙槽后神经阻滞麻醉.docx",
    AuditType.MAXILLARY_IMPRESSION:        "video/上颌印模制取/上下牙列印模制取.docx",
    AuditType.IA_NERVE_BLOCK:              "video/下牙槽神经阻滞麻醉/下牙槽神经阻滞麻醉.docx",
    AuditType.SUBMANDIBULAR_GLAND_EXAM:    "video/下颌下腺检查/下颌下腺检查.docx",
    AuditType.INTRAORAL_SUTURE:            "video/口内缝合/口内缝合术.docx",
    AuditType.PERCUSSION_EXAM:             "video/叩诊检查/一般检查-叩诊.docx",
    AuditType.POSTERIOR_CROWN_PREP:        "video/后牙铸造全冠牙体预备/后牙铸造全冠的牙体预备.docx",
    AuditType.OXYGEN_THERAPY:              "video/吸氧术/吸氧术.docx",
    AuditType.OCCLUSION_EXAM:              "video/咬合关系检查/咬合关系检查.docx",
    AuditType.CPR:                         "video/心肺复苏/心肺复苏.docx",
    AuditType.PALPATION_EXAM:              "video/扪诊检查/一般检查-扪诊.docx",
    AuditType.PROBING_EXAM:                "video/探诊检查/一般检查-探诊.docx",
    AuditType.MODIFIED_BASS_BRUSHING:      "video/改良BASS刷牙法/改良BASS刷牙法.docx",
    AuditType.MODIFIED_CPI_PROBING:        "video/改良CPI探诊检查/改良社区牙周指数检查及记录.docx",
    AuditType.MOBILITY_EXAM:               "video/松动度检查/一般检查-松动度.docx",
    AuditType.RUBBER_DAM_ISOLATION:        "video/橡皮障隔离术/橡皮障隔离术.docx",
    AuditType.PERIODONTAL_PROBING:         None,  # COS 目录为空，暂无文件
    AuditType.TMJ_EXAM:                    "video/颞下颌关节检查/颞下颌关节检查.docx",
    AuditType.TOOTH_EXTRACTION:            "video/牙拔除术/牙拔除术.docx",
    AuditType.FLOSS_GUIDANCE:              "video/牙线使用指导/牙线使用指导.docx",
    AuditType.PULP_VITALITY_TEST:          "video/牙髓活力测试/牙髓活力测试.docx",
    AuditType.MOLAR_ACCESS_OPENING:        "video/磨牙开髓术/开髓术.docx",
    AuditType.EXTRACTED_MOLAR_CAVITY_PREP: "video/离体磨牙复面洞制备术/离体磨牙复面洞制备术.docx",
    AuditType.PIT_FISSURE_SEALING:         "video/窝沟封闭/窝沟封闭操作流程.docx",
    AuditType.ABSCESS_INCISION_DRAINAGE:   "video/脓肿切开引流术/牙槽脓肿切开引流.docx",
    AuditType.MUCOSA_DISINFECTION:         "video/黏膜消毒/口腔黏膜消毒.docx",
    AuditType.SUPRAGINGIVAL_SCALING:       "video/龈上洁治/龈上洁治术.docx",
    AuditType.BLOOD_PRESSURE_MEASUREMENT:  "video/血压测量/测量血压.docx",
    AuditType.VISUAL_EXAM:                 "video/视诊检查/一般检查-视诊.docx",
}


def get_score_doc_path(audit_type: AuditType, local_dir: str = "doc/score_doc") -> str | None:
    """
    根据审核类型获取本地 docx 评分文档路径

    :param audit_type: 审核类型枚举
    :param local_dir: 本地文档目录，默认 doc/score_doc
    :return: 本地文件路径，无文件时返回 None
    """
    cos_key = AUDIT_SCORE_DOC.get(audit_type)
    if cos_key is None:
        return None
    filename = cos_key.rsplit("/", 1)[-1]
    return str(BASE_DIR / local_dir / filename)


# 提取后的纯文本规则目录
SCORE_RULES_DIR = BASE_DIR / "doc" / "score_rules"


def get_score_rule_text(audit_type: AuditType) -> str | None:
    """
    获取审核类型对应的打分规则文本（从预提取的 txt 文件中读取）

    :param audit_type: 审核类型枚举
    :return: 打分规则纯文本，无文件时返回 None
    """
    doc_path = get_score_doc_path(audit_type)
    if doc_path is None:
        return None
    # docx 文件名 → 同名 txt 文件
    txt_path = SCORE_RULES_DIR / (Path(doc_path).stem + ".txt")
    if not txt_path.exists():
        return None
    return txt_path.read_text(encoding="utf-8")
