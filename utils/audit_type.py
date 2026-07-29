# @Auther  : luoscorn
# @Time    : 2026/7/29: 16:51
# @File    : audit_type.py
from enum import Enum


class AuditType(str, Enum):
    """审核类型枚举 值与 cos 桶 video/oss 下的目录名一一对应"""
    PSA_NERVE_BLOCK = "上牙槽后神经阻滞麻醉"
    MAXILLARY_IMPRESSION = "上颌印模制取"
    IA_NERVE_BLOCK = "下牙槽神经阻滞麻醉"
    SUBMANDIBULAR_GLAND_EXAM = "下颌下腺检查"
    INTRAORAL_SUTURE = "口内缝合"
    PERCUSSION_EXAM = "叩诊检查"
    OCCLUSION_EXAM = "咬合关系检查"
    PALPATION_EXAM = "扪诊检查"
    PROBING_EXAM = "探诊检查"
    MODIFIED_BASS_BRUSHING = "改良BASS刷牙法"
    MODIFIED_CPI_PROBING = "改良CPI探诊检查"
    MOBILITY_EXAM = "松动度检查"
    RUBBER_DAM_ISOLATION = "橡皮障隔离术"
    PERIODONTAL_PROBING = "牙周探诊"
    TOOTH_EXTRACTION = "牙拔除术"
    FLOSS_GUIDANCE = "牙线使用指导"
    PULP_VITALITY_TEST = "牙髓活力测试"
    PIT_FISSURE_SEALING = "窝沟封闭"
    ABSCESS_INCISION_DRAINAGE = "脓肿切开引流术"
    BLOOD_PRESSURE_MEASUREMENT = "血压测量"
    VISUAL_EXAM = "视诊检查"
    TMJ_EXAM = "颞下颌关节检查"
    MUCOSA_DISINFECTION = "黏膜消毒"
    SUPRAGINGIVAL_SCALING = "龈上洁治"
