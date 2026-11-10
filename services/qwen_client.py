# @Auther  : luoscorn
# @Time    : 2026/7/30
# @File    : qwen_client.py
"""千问 VL 视频打分客户端 — 支持整段视频与「切片关键帧 + 语音转写」两种输入"""

import base64
import io
import json
import logging
import re
import time
from typing import Optional

import requests
from openai import OpenAI

try:
    from PIL import Image, ImageOps
except ImportError:      # 镜像未重建(缺 Pillow)时退化为不压缩 不阻断打分主流程
    Image = ImageOps = None

from config import settings
from services.media_evidence import VideoEvidence, frames_payload
from utils.oss_client import get_object_url
from utils.score_rubric import RubricItem

logger = logging.getLogger(__name__)

# 进程内复用的客户端 首次调用时创建
_client: Optional[OpenAI] = None

# 千问 OpenAI 兼容端点
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 模型名走配置 供应商下线模型时改 .env 即可(qwen-vl-max 已于 2026-10 下线)
_MODEL = settings.QWEN_VL_MODEL

# 单次请求超时 180s 且不让 openai 内部重试: 重试权由 audit_worker 统一掌握
# openai 默认 timeout 600s + max_retries 2 会与 worker 的 3 次重试叠加 把单任务拖到小时级并占住并发名额
_REQUEST_TIMEOUT = 180.0
# 缺项补问轮数 与 补问仍缺时的兜底给分比例(取证据受限档下限 60%)
_MAX_REPAIR_ROUNDS = 1
_MISSING_SCORE_RATIO = 0.6
_MISSING_ITEM_REASON = "模型未返回该评分项 判定缺失 按证据受限档下限暂记 需人工复核"
# summary 长度上限 与 _SCORING_POLICY 声明的 60 字保持一致
_SUMMARY_MAX_CHARS = 60
# 规范参考图: 单项取几张与整单总张数上限
# 同一评分点常配多张(不同角度/正误对照) 只取首图会丢掉一半判断依据
# 总量仍要扣着算: 叠加切片关键帧后单请求图片总数不宜再涨
_MAX_REFERENCE_IMAGES_PER_ITEM = 3
_MAX_REFERENCE_IMAGES = 12
# 参考图下载超时 与 进程内缓存时长(参考图随规则版本固定 改图会换 key)
_REFERENCE_TIMEOUT = 15.0
_REFERENCE_CACHE_TTL = 600
_REFERENCE_CACHE_LIMIT = 200
# 参考图压缩: 长边上限高于关键帧(1024)以保留规范细节 仅在超长边或超体积时重编码
# 阈值内的图原样透传 不做任何重编码 避免无谓的细节损失
_REFERENCE_MAX_EDGE = 1280
_REFERENCE_MAX_BYTES = 400 * 1024
_REFERENCE_JPEG_QUALITY = 92
_reference_cache: dict[str, tuple[float, str]] = {}


# 模型偶发用 1 / (1) / ①) 这类写法 统一归到 ① 形式再与评分项匹配
_NUMBER_ALIASES = {
    str(idx): circled
    for idx, circled in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", start=1)
}

# 评分尺度: 按评分表严格判定 该扣必扣 但不因证据受限误判为未执行
_SCORING_POLICY = """【评分尺度：按评分表严格判定 该扣必扣 但不因证据受限误判】
0. 优先级：本次给出的评分表是最终判定口径，高于本文件其余通用条文。每一项的「评分点」「视频识别依据」「备注」共同构成该项的判定标准：
   - 规范参考图只帮助理解，不构成额外要求：考生用的器械、动作姿态、手部位置与参考图不一致本身不是扣分理由，判定只看是否达成评分点文字描述的"效果"。
   - 表格或备注已写明达成条件、判定方式或给分档位时，一律以表格为准，即使与下列通用梯度不同（例如某项备注写明“画面中能看到支点即视为达标给满分”，就按满分给，不再以“支点偶有悬空”扣分）。
   - 表格只描述正确做法、未写明判定口径时，才按下述梯度判定。
1. 逐项独立判定：先确认画面或语音中是否存在该评分点的证据，再决定给分；既不因“大胆做过”给虚高分，也不因“没看清”直接判未执行。
2. 给分梯度：从上往下判定 命中即停 不叠加
   - 口述完整（把动作要素说清），且该项未填口述视觉权重或权重全部归口述 → 该项满分，不论画面是否清晰呈现；reason 引用口述原文，画面有佐证时一并引用
   - 动作可见但不规范、不到位或不完整 → 60%~85%
   - 口述简略但画面有相关动作 → 60%~85%
   - 证据受限且无相反证据（画面未拍到、遮挡、处于镜头外，口述也未交代） → 60%~85%，reason 注明“证据受限 未见违规”
   - 口述简略且画面无相关动作 → 最高 50%
   - 明确错误、与要求相反，或同一评分点内口述与画面矛盾 → 以画面为准，0~20%
   - 整项未执行（所有片段的画面与口述都无任何迹象）→ 0
3. 以下情形必须扣分，不得豁免：器械选择错误、操作顺序颠倒、跨越无菌区、器械污染、明确缺少支点或支点不稳、明显超出或绕过规定检查范围、次数或时长明显不达标、有伤害性动作。
4. 判定原则
   - 口述视觉权重是选填项：某项没有给出口述视觉权重时，不做证据来源的份额折算，只要该项符合评分点、视频识别依据与备注写明的要求就给该项满分，不得因为“只有画面达成、口述未交代”或“只有口述说清、画面未呈现”而按比例打折。
   - 某项填写了口述视觉权重时，它讲的是该项达成证据的来源分工，不是给分上限，且优先于第 2 条第 1 档：权重写明的几份证据都达成即合计达标，按该项满分给，不得只按其中一份打折后交差；只达成其中一份时才按该份对应的比例折算。权重里没有口述份额（如只写视觉）时，口述再完整也只作佐证，不能单凭口述给分，按画面判定，画面未拍到时按证据受限档；权重全部归口述时按口述判定。reason 只写达成了哪几份证据，不写算式。
   - 口述完整：把该评分点的动作要素说清（用什么器械、在哪个部位、怎么做、观察什么、判断结果如何）；只提了步骤名称或关键词属口述简略，按第 2 条对应档位给分。
   - 口述与画面分开识别、各自归属：口述内容归到它描述的评分点，画面动作归到它实际对应的评分点。同一时刻口述在说 A 而画面在做 B，是两个不同评分点的证据，不构成矛盾，不得据此扣 A 的分，也不得把 B 计到 A 上。
   - 只有针对同一个评分点、口述与画面相互排斥时才算矛盾（如该项口述说用大弯端，画面清楚显示用的是三弯端），以画面为准给 0~20%，reason 同时写出“口述原文”与画面依据。
   - 口述提到某项动作而画面没有拍到，不构成矛盾：这两者是同一评分点的不同证据来源，按本条口述视觉权重的说明判定即可，禁止写成“口述与画面矛盾”“以画面为准”并据此判 0。
   - 各项判定以有经验考官现场评分会给出的结论为基准：不因照顾学员感受把未达标写成满分，也不因拍摄条件局限把合格操作判为未执行。
   - 否定性结论必须给出反证：确认为未执行或给低分时，reason 必须写出关键帧里与该评分点直接冲突的具体画面；若只是画面未拍到、角度欠佳、细节模糊，一律表述为「证据受限」并按证据受限档给分，禁止写成「未使用」「未执行」「未做到」「不稳定」。
   - 模型牙/仿真头模是标准考核场景：不得以「不是在患者口腔内」「模型无牙龈、无唾液、无真实口角」为由扣分或降档，只按动作在模型上是否规范完成判定。
   - 持续型姿态按出现过即认定：口镜牵拉口角、无名指支点、执笔式握持这类是维持性姿态，只要若干关键帧中出现过该姿态即认定达成，不要求看到动作全过程；相邻帧之间姿态的短暂变化不构成支点不稳。
   - 每项只按该项评分点的字面要求判定，不得把其他评分项的要求挪作本项的扣分理由：例如「口镜牵拉口角」这项才要求口镜实际牵拉，而「左手持口镜、右手持探针」这类持械项只要求两手各持对应器械，全程未牵拉也不影响本项给满分。
   - 同一处缺陷只在它对应的评分点扣一次分，不得跨项重复扣：例如未牵拉口角已在「口镜牵拉口角」项扣分，就不得再以此为由扣「持械分工」项的分；每项 reason 只引用与本项评分点直接相关的证据。
5. summary 只写总体结论：用一两句话概括整体完成情况与主要失分方向（不超过 60 字），不逐条罗列失分点、不复述各扣分明细（细节放在对应 items 的 reason 里），不使用表扬、鼓励式表述，也不写空泛套话。"""

# 器械与姿态的专属认定口径: 只有评分项里提到相关字眼时才写进提示词
# 27 个审核类型共用同一份评分尺度 心肺复苏、测血压读到「口镜与牙科镊的区别」只会稀释注意力
_ITEM_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("牵拉", "口角"),
     "口镜牵拉只看镜面是否到位：画面中口镜工作端（镜面）抵住或撑开一侧口角、颊侧黏膜，即认定「口镜牵拉口角」达成，不要求看到软组织明显变形（牙模型口角不易形变），也不要求全程持续。"),
    (("支点",),
     "支点的认定：握持器械那只手的无名指或中指抵在邻牙、对侧牙、模型基座、下颌或患者面部任何部位，形成对抗即为有效支点。不要求手指悬空出明显的三角、不要求压在特定牙位，也不要求该指始终不被遮挡；不得因为支点落在画面边缘、被手背或器械挡住一部分、或只在部分帧可见就判定无支点。"),
    (("口镜", "持械"),
     "手部持械分工只看两手是否各持不同器械：口镜的形态特征是长柄加末端镜面圆盘，长度明显长于探针，镜面通常朝下对着口角；只要某只手里握的是这种长柄圆头器械即认定为口镜，不要求镜面反光清晰可见，也不要求该手此刻正在牵拉。器械颜色与型号不作要求：口镜常为全白色，与白手套、白工作服同色，此时画面里往往只能看到一根比探针更粗更长的白色手柄和末端白色圆盘背面，甚至镜面背对镜头只露出一片扁平圆盘，这仍然算口镜；不得因为与参考图里的器械颜色不同、或看不到镜面反光、或圆盘面积小而否认它是口镜。判断长柄还是细针要比对两者的整段可见长度，不要把被手握住的那截排除在外后只比探出头的一小段。镜头存在镜像、左右判读有歧义时，只要两手各持不同器械（一为长柄圆头、一为两端弯曲的细探针）即视为分工明确，不得仅因左右手判反、或口镜用途未逐帧可见而判该项未执行。"),
    (("口镜", "镊"),
     "口镜与牙科镊的区别：镊子末端是两片可张合的尖刃或齿状夹持片，末端没有盘面；口镜末端是**一整片扁平圆形或椭圆盘**，盘面与手柄成一定夹角，镜面因反射铺巾、灯光或工作服常呈灰蓝、灰白或高亮色而非典型银色。画面里只要看到「细长手柄 + 末端一整片扁平圆盘」，就判定为口镜，不得因为反光弱、颜色发灰、看不清反射内容、或把盘面边缘看成两片而改判为牙科镊、棉镊。"),
)


def _scoring_policy(rules_text: str) -> str:
    """评分尺度 + 本次评分项涉及的器械与姿态认定口径"""
    picked = [rule for keys, rule in _ITEM_RULES if any(key in (rules_text or "") for key in keys)]
    if not picked:
        return _SCORING_POLICY
    lines = [_SCORING_POLICY, "", "【本次评分项涉及的器械与姿态认定】以下口径与评分尺度第 4 条同级："]
    lines += [f"   - {rule}" for rule in picked]
    return "\n".join(lines)

# 切片输入的证据说明
_EVIDENCE_NOTE = """【证据说明】本次输入是按时间轴组织的视频关键帧图片 + 对应片段的语音转写文本，不是完整视频。
关键帧取自画面变化最明显的时刻（换器械、换部位、开始/结束动作），标签中的 @mm:ss 是该帧的准确时刻。口述内容与画面动作要分开归属到各自的评分点，再只看同一评分点内两者是否相互排斥，不要因为同一时刻口述与画面在别的事上就判矛盾，也不要跳片段混用证据。
关键帧数量有限，动作可能落在两帧之间、被遮挡或处于镜头外：口述已把动作要素说清且第 2 条第 1 档的前提成立时仍给满分；口述也未交代时才算证据受限，按评分尺度的证据受限档给分并在 reason 注明“证据受限”，任何情况下都不得据此判为未执行。
口述（语音转写）与画面同为完成依据：未填口述视觉权重或权重全部归口述的项，口述把动作要素说清即给满分，不因画面未清晰呈现扣分；填写了视觉份额的项按评分尺度第 4 条的来源分工判定。只有同一评分点内口述与画面相互排斥时，才以画面为准。
某片段标注「语音转写失败」时，表示该段口述没有被转写出来，不代表考生没说话：不得据此认定口述未交代，涉及口述的评分项按证据受限档处理。"""

# 规范参考图说明: 图来自评分标准的正确操作示意 只能当规范参照 不能当考生证据
_REFERENCE_NOTE = """【规范参考图说明】下面先给出若干张来自评分标准的操作示意图，只用于帮助你理解该项评分点在考核什么、什么算规范，不是标准答案模板。
同一评分点可能配有多张参考图（不同角度、不同部位、或规范与不规范的对照）：标签写成「参考图 编号-序号」的几张属于同一个评分点，要合起来理解该项的规范要求，不能因为其中一张没体现某项要素就当作没配图。
参考图里的器械与动作都只是示意：器械的颜色、材质、型号、长短可能不同（参考图可能是彩色手柄大镜面，考生手里可能是全白色细长手柄），握持姿势、手部位置、牵拉角度、动作顺序与节奏也可能不同。只要考生达成了该项「视频识别依据」里描述的效果就算达标，不要求器械外观与参考图一致，也不要求动作姿态与参考图逐帧吻合，更不得因为考生用的器械与参考图长得不一样就判定未使用该类器械。判定锚点是评分点的文字要求，不是与参考图的相似度。
参考图不是考生本人的操作证据：不得因为某项配了参考图就推定考生已完成该评分点，也不得把参考图里的画面当作考生画面引用。每一项都必须依据考生视频的关键帧与语音转写判定。"""

# 要求 AI 返回的 JSON 结构说明
_OUTPUT_FORMAT = """请严格按照以下 JSON 格式返回打分结果，不要输出任何多余内容：
{
  "full_score": <float, 满分，根据评分规则中所有评分项分值之和>,
  "total_score": <float, 实际得分，必须等于 full_score 减去所有 deduct_score 之和>,
  "deductions": [
    {
      "item": "<扣分项名称，对应评分规则中的具体步骤>",
      "deduct_score": <float, 该项扣分数>,
      "reason": "<扣分原因，说明操作不规范之处>"
    }
  ],
  "summary": "<总体结论，一两句话概括完成情况与主要失分方向，不逐条罗列失分点，不超过 60 字>"
}
重要：total_score 必须严格等于 full_score - sum(deduct_score)，请仔细核对计算。
各项均达标时 deductions 为空数组，total_score 等于 full_score；只要有评分点未达标就必须列出对应扣分。"""

# 结构化评分模式：AI 只按评分项给分 总分与扣分项由服务端汇总
_RUBRIC_OUTPUT_FORMAT = """请严格按照以下 JSON 格式返回评分结果，不要输出任何多余内容：
{
  "items": [
    {
      "number": "<评分项编号，如 ①>",
      "score": <float, 该项实际得分>,
      "time": "<该评分点证据所在时刻，格式 mm:ss 或 mm:ss-mm:ss，必须取自关键帧标签中的 @mm:ss；全片无任何迹象填 无>",
      "reason": "<判定理由，说明画面或语音中可见/可闻的依据>"
    }
  ],
  "summary": "<总体结论，一两句话概括完成情况与主要失分方向，不逐条罗列失分点，不超过 60 字>"
}
要求：
1. items 必须覆盖评分规则中的全部评分项，每个编号恰好出现一次，不得遗漏、重复或编造编号。
2. score 范围为 0 到该项分值；先按该评分项正文与备注给出的判定口径给分，未写明时按评分尺度的给分梯度判定，不得为凑分给出虚高分数。
3. 不要自行计算总分，服务端会根据各项得分汇总。
4. reason 必须引用口述原文或画面中的具体依据，不能只写“未做到”“不规范”等空泛描述。
5. reason 用陈述句直接写判定依据与结论，不超过 100 字；分数只由 score 字段表达。禁止出现算式、分值换算（如 0.2×0.8=0.16）、“取上限/取下限”、百分比档位描述（如“给60%”“按80%给分”），以及任何思考过程：反问句、“但按规则应”“再核一遍”“不，应该是”“故应得”这类自我辩论和改口重判，一律不得写进 reason。reason 只写一次最终结论，如“@00:26 右手无名指抵住模型下颌基座，口述提及支点，达标”，不写你是如何一步步想出来的。
6. time 用于证据受限项的二次复核定位，必须直接引用关键帧标签里的 @mm:ss，不得自行推算；动作持续一段时写起止两个时刻，无法定位填 无。
7. summary 不重复 items 里的扣分点，只给总体完成度式的结论。"""

# reason 里模型常附带分数换算过程 统一剔除 只保留判定依据
# 例: “给60%即0.12分”、“按80%×0.1=0.08分”、“四舍五入为0.15分”、“扣0.05分”、“得0.14”
_REASON_CALC_RE = re.compile(
    r"[，,；;：:]?[（(]?\s*(?:按|给|取|即)?\s*\d+(?:\.\d+)?\s*%?\s*(?:[×xX*]|=)\s*\d+(?:\.\d+)?\s*%?[^，,；;。）)]*"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:即|给)?\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:给|计)分"
    r"|[，,；;：:]?\s*四舍五入为?\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*扣\s*\d+(?:\.\d+)?\s*分"
    r"|[，,；;：:]?\s*得\s*\d+(?:\.\d+)?\s*分?\s*$"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*[~～〜-]\s*\d+(?:\.\d+)?\s*%\s*(?:给|计)?\s*分?"
    r"|[，,；;：:]?\s*(?:按|给|取)\s*\d+(?:\.\d+)?\s*%\s*(?:即|给|计)?\s*分?"
)


# reason 里出现这些说明模型在写推理过程而不是判定依据 从首个命中处砍断
# 实测原句: “口述占20%、视觉占80%，两项均满足，但按规则应直接给满分0.2？不，规则明确……”
_REASON_REASONING_RE = re.compile(
    r"[，,；;]?\s*(?:"
    r"但(?:按|依|根据)?规则|"
    r"不[，,]\s*(?:规则|按|而)|"
    r"再(?:核|审|确认|算|检)|"
    r"重新(?:核|算|判)|"
    r"按规则应|"
    r"应(?:直接)?给(?:满分|分)|"
    r"应得|"
    r"故(?:应|得|给)|"
    r"注[：:]|"
    r"实际得分|"
    r"给满分|"
    r"折合|"
    r"score\s*[=＝]|"
    r"综上所述|"
    r"但为严谨"
    r")"
)

# reason 字数上限 与输出格式里声明的限制保持一致
_REASON_MAX_CHARS = 100


def _outside_quotes(text: str, index: int) -> bool:
    """index 处是否在引号外 reason 常引用考生的原话 引号里的问句是证据不是自问自答"""
    head = text[:index]
    depth = sum(head.count(mark) for mark in "“「『") - sum(head.count(mark) for mark in "”」』")
    return depth <= 0 and head.count('"') % 2 == 0


def _trim_reasoning(text: str) -> str:
    """砍掉 reason 里模型写出来的推理过程

    这类文本对判定没有价值 而且会在补证阶段作为「上一轮判定」喂回模型变成噪声
    """
    if not text:
        return ""
    # 反问几乎只出现在自我辩论里 正常判定不会用问句
    # 但引号里的问句是考生原话(“有没有不舒服？”) 在那里截断会连后半句的结论一起丢掉
    for index, ch in enumerate(text):
        if ch in "？?" and index > 0 and _outside_quotes(text, index):
            text = text[:index]
            break
    match = _REASON_REASONING_RE.search(text)
    if match:
        text = text[:match.start()]
    # 砍断处可能停在时刻中间 去掉尾部残数字
    text = re.sub(r"[0-9:：\-~至到]+$", "", text)
    return text.strip(" ，,；;：:")


def _strip_unpaired_brackets(text: str) -> str:
    """删除无法配对的括号 保留配对完整的括号内容"""
    drop: set[int] = set()
    stack: list[int] = []
    for idx, ch in enumerate(text):
        if ch in "（(":
            stack.append(idx)
        elif ch in "）)":
            if stack:
                stack.pop()
            else:
                drop.add(idx)
    drop.update(stack)
    return "".join(ch for idx, ch in enumerate(text) if idx not in drop)


def _clean_reason(reason: str) -> str:
    """去掉 reason 中的分数换算与推理过程 只保留判定依据"""
    if not reason:
        return ""
    # 自问自答与再核算属于思考过程 不属于判定依据
    trimmed = _trim_reasoning(reason)
    cleaned = _REASON_CALC_RE.sub("", trimmed)
    # 算式被剔除后可能残留空括号或落单括号 一并清理
    cleaned = re.sub(r"[（(]\s*[）)]", "", cleaned)
    cleaned = _strip_unpaired_brackets(cleaned)
    cleaned = re.sub(r"[，,；;：:]{2,}", "，", cleaned).strip(" ，,；;：:")
    # 清洗后为空且原文本身就是分数换算或推理 则不再退回原文 避免把脏文本当日判定依据
    if not cleaned:
        hit_calc = _REASON_CALC_RE.search(reason)
        return "" if (hit_calc or trimmed != reason.strip(" ，,；;：:")) else reason
    if len(cleaned) > _REASON_MAX_CHARS:
        cleaned = cleaned[:_REASON_MAX_CHARS].rstrip(" ，,；;：:") + "…"
    return cleaned


# reason 已明确写出「达成」的结论 而 score 仍按份额打折
# 实测原句: “视觉占80%达成、口述20%补充，合计达标” 同一条却只给 0.16
# 实测原句: “口述占20%、视觉占80%，画面清晰可见三弯端用于邻面，达标”
_REASON_FULL_MARK_RE = re.compile(
    r"合计达标|均达成|均满足|均符合|均支持|全部达成|完全达成|两项均|两者均|各项均|"
    r"(?:视觉|画面)[^，,。；;]{0,14}?口述[^，,。；;]{0,14}?达成"
)
# 结论落在句尾也算 模型常把“达标”“满分”放在最后 前面只是证据叙述
# 实测原句: “...握持姿势稳定，探针控制良好，无滑脱抖动，视觉100%规范，满分”
_REASON_DONE_TAIL_RE = re.compile(r"(?:达标|达成|满足|符合要求|通过|满分)[。．.!！\s]*$")
# 带有否定或限定词时不做上修 否则“合计达标但支点不稳”“部分牙位达标”会被抬成满分
_REASON_FAIL_MARK_RE = re.compile(
    r"未达|不达标|未满足|未执行|未完成|未能|未做到|缺失|未见|无任何|不符合|不规范|不到位|不稳|不足"
)
_REASON_HEDGE_RE = re.compile(r"部分|基本|大体|个别|局部|仅|但|不过|尚未")


# 反方向的错配: 分数填了满分 reason 的结论却是没做到
# 实测补证原句: “口镜仍在托盘内未被拿起；两手未各持口镜与探针，分工不符要求” 同一条却 clear=true、满分
# 只认结论性措辞: 「未见戳刺牙龈」这类正面描述里的否定词不算
# 「无法确认」也不算: 口述完整时第 1 档允许画面无法确认仍给满分
_REASON_VERDICT_FAIL_RE = re.compile(
    r"不符(?:合)?(?:要求|标准|规范)|未达标|不达标|未满足|未执行|不得分|"
    r"两手未各持|未见[^。；;]{0,50}分工(?:状态|要求)?|"
    r"(?:左手|右手)未持(?:口镜|任何器械)"
)


def _contradicts_full_mark(score: float, reason: str, full_score: float) -> bool:
    """满分与 reason 的否定结论互相矛盾 这样的分数不可信 不得据此抬分"""
    if full_score <= 0 or score < full_score - 0.001:
        return False
    return bool(_REASON_VERDICT_FAIL_RE.search(reason or ""))


def _align_score_with_reason(score: float, reason: str, full_score: float) -> tuple[float, bool]:
    """reason 已写明达成时 把打折的分数提回满分

    模型常把结论写成「视觉达成、口述也把要素说清」 手却只填了满分×视觉份额
    分数只由 score 字段表达 结论与分数打架时必须让分数服从结论

    :return: (分数, 是否修正)
    """
    if full_score <= 0 or not reason or score >= full_score - 0.001:
        return score, False
    if _REASON_FAIL_MARK_RE.search(reason) or _REASON_HEDGE_RE.search(reason):
        return score, False
    if not (_REASON_FULL_MARK_RE.search(reason) or _REASON_DONE_TAIL_RE.search(reason)):
        return score, False
    return round(float(full_score), 2), True


def _inline(text: str) -> str:
    """把多行文本压成单行 备注允许换行录入 但进提示词必须是单行"""
    return " ".join(str(text or "").split())


def _normalize_number(number) -> str:
    """把 1 / (1) / 1) 这类编号归一到 ① 形式"""
    text = str(number or "").strip()
    core = re.sub(r"[()（）\s.]", "", text)
    return _NUMBER_ALIASES.get(core, text)


_SUMMARY_LIST_RE = re.compile(r"(?:(?<=^)|(?<=[\s，,；;。]))(?:\d+[.、)）]|[①-⑳]|[-•·])\s*")


def _clean_summary(summary: str) -> str:
    """summary 兜底: 去列表符号与换行 并按 60 字截断"""
    text = _inline(summary)
    text = _SUMMARY_LIST_RE.sub("", text)
    text = re.sub(r"[，,；;、]{2,}", "，", text).strip(" ，,；;、")
    if len(text) > _SUMMARY_MAX_CHARS:
        text = text[:_SUMMARY_MAX_CHARS].rstrip(" ，,；;、") + "…"
    return text


def _get_client() -> OpenAI:
    """获取进程内复用的千问客户端 复用连接池避免每次调用重建 TLS"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.QWEN_API_KEY_1,
            base_url=_BASE_URL,
            timeout=_REQUEST_TIMEOUT,
            max_retries=0,
        )
    return _client


def _build_system_prompt(
    audit_type: str,
    rules_block: str,
    output_format: str,
    evidence: Optional[VideoEvidence],
) -> str:
    """拼装系统提示词: 评分规则 + 严格评分尺度 + 证据说明 + 输出格式"""
    parts = [
        "你是一位严格、客观的口腔医学操作考核评分专家，只依据评分表与视频证据判定，不放宽标准。",
        f"请根据以下评分规则，对「{audit_type}」操作进行逐项打分。",
        "",
        rules_block,
        "",
        _scoring_policy(rules_block),
    ]
    if evidence is not None:
        parts += ["", _EVIDENCE_NOTE]
    parts += ["", "【输出要求】", output_format]
    return "\n".join(parts)


def _shrink_image(data: bytes, mime: str) -> tuple[bytes, str]:
    """规整参考图: 修正 EXIF 方向 超长边或超体积时等比缩小并转高质量 JPEG

    阈值内的图原样返回 不重编码 保证规范细节不被压缩损失
    """
    if Image is None:
        return data, mime
    try:
        with Image.open(io.BytesIO(data)) as probe:
            long_edge = max(probe.size)
            exif = probe.getexif()
            rotated = bool(exif and exif.get(0x0112, 1) != 1)
        # 手机直拍的图常带 EXIF 旋转 不改正会让模型看到躺倒的操作示意
        if long_edge <= _REFERENCE_MAX_EDGE and not rotated and len(data) <= _REFERENCE_MAX_BYTES:
            return data, mime

        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            if "A" in img.getbands():
                # JPEG 不支持透明通道 透明区铺白底 避免转档后出现黑块
                rgba = img.convert("RGBA")
                canvas = Image.new("RGB", rgba.size, (255, 255, 255))
                canvas.paste(rgba, mask=rgba.getchannel("A"))
                img = canvas
            else:
                img = img.convert("RGB")
            # thumbnail 只缩不放 长边压到阈值内
            img.thumbnail((_REFERENCE_MAX_EDGE, _REFERENCE_MAX_EDGE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(
                buf, format="JPEG", quality=_REFERENCE_JPEG_QUALITY, subsampling=1, optimize=True
            )
            size = img.size
        shrunk = buf.getvalue()
    except Exception:
        logger.warning("参考图压缩失败 改用原图", exc_info=True)
        return data, mime
    # 极端情况下重编码反而更大 仍用原图
    if len(shrunk) >= len(data):
        return data, mime
    logger.info(
        "参考图已压缩: %dKB -> %dKB %dx%d", len(data) // 1024, len(shrunk) // 1024, size[0], size[1]
    )
    return shrunk, "image/jpeg"


def _image_data_uri(ref: str) -> str:
    """操作图片转 data URI: OSS key 先预签名再下载 http 直链直接下载 失败返回空串

    与关键帧一样用 base64 内联 不把外链交给模型去拉 避免签名过期或模型侧抓取失败
    """
    cached = _reference_cache.get(ref)
    if cached is not None and time.time() - cached[0] < _REFERENCE_CACHE_TTL:
        return cached[1]
    try:
        url = ref if ref.startswith(("http://", "https://")) else get_object_url(oss_key=ref, expired=3600)
        resp = requests.get(url, timeout=_REFERENCE_TIMEOUT)
        resp.raise_for_status()
        mime = (resp.headers.get("content-type") or "").split(";")[0].strip()
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        data, mime = _shrink_image(resp.content, mime)
        data_uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        logger.warning("规范参考图获取失败 已跳过: %s", ref, exc_info=True)
        return ""
    if len(_reference_cache) >= _REFERENCE_CACHE_LIMIT:
        _reference_cache.clear()
    _reference_cache[ref] = (time.time(), data_uri)
    return data_uri


def _reference_images(rubric_items: list[RubricItem]) -> list[tuple[str, str]]:
    """收集各评分项的规范参考图

    单项最多 _MAX_REFERENCE_IMAGES_PER_ITEM 张 整单受 _MAX_REFERENCE_IMAGES 约束
    多图项的标签带序号(参考图 3-1/3-2) 让模型知道这几张属于同一评分点

    按轮次分配: 先给每项首图 再逐项补第二张 第三张。若按项顺序先到先得
    前面的项会把整单名额占满 后面的项一张参考图都拿不到 且悄无声息
    """
    plans = [
        (item, item.images[:_MAX_REFERENCE_IMAGES_PER_ITEM]) for item in rubric_items
    ]
    images: list[tuple[str, str]] = []
    for round_idx in range(_MAX_REFERENCE_IMAGES_PER_ITEM):
        if len(images) >= _MAX_REFERENCE_IMAGES:
            break
        for item, refs in plans:
            if len(images) >= _MAX_REFERENCE_IMAGES:
                break
            if round_idx >= len(refs):
                continue
            data_uri = _image_data_uri(refs[round_idx])
            if not data_uri:
                continue
            label = (
                f"参考图 {item.number}"
                if len(refs) == 1
                else f"参考图 {item.number}-{round_idx + 1}"
            )
            images.append((label, data_uri))
    return images


def _prompt_reference_images(rubric_items: list[RubricItem]) -> list[tuple[str, str]]:
    """首轮实际送给模型的参考图 由 MEDIA_REFERENCE_IMAGES_ENABLED 控制

    参考图在提示词里只是示意 已经明确不参与判定 却和证据帧争夺同一份注意力预算
    默认关闭 需要时在 env 里打开
    """
    if not settings.MEDIA_REFERENCE_IMAGES_ENABLED:
        return []
    return _reference_images(rubric_items)


def _build_user_content(
    video_url: str,
    evidence: Optional[VideoEvidence],
    instruction: str,
    reference_images: Optional[list[tuple[str, str]]] = None,
) -> list[dict]:
    """构造用户消息: 规范参考图 + 证据(切片关键帧+语音 或 整段视频) + 判分指令"""
    content: list[dict] = []
    if reference_images:
        # 参考图放在考生证据之前 并显式声明非考生证据 避免模型把规范图当成本次操作画面
        content.append({"type": "text", "text": _REFERENCE_NOTE})
        for label, url in reference_images:
            content.append({"type": "text", "text": f"{label}："})
            content.append({"type": "image_url", "image_url": {"url": url}})
    if evidence is None:
        content.append({"type": "video_url", "video_url": {"url": video_url}})
    else:
        # 先给时间轴与语音全文 再按片段顺序贴关键帧
        content.append({"type": "text", "text": evidence.timeline_text()})
        for label, frame in frames_payload(evidence):
            content.append({"type": "text", "text": f"{label}："})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
            })
    content.append({"type": "text", "text": instruction})
    return content


# 默认采样温度: 同一模型同一证据仍会给出相反结论 投票时靠拉开温度增加样本差异
_DEFAULT_TEMPERATURE = 0.1


def _call_model(
    system_prompt: str,
    user_content: list[dict],
    audit_type: str,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> str:
    """调用千问多模态接口并返回原始文本

    :param temperature: 采样温度 投票时拉开取值以拿到互相独立的判定
    """
    client = _get_client()
    options = {}
    if settings.QWEN_ENABLE_THINKING:
        # 思考模式把推理过程放进 reasoning_content 不再挤在 reason 里自问自答
        options["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": settings.QWEN_THINKING_BUDGET,
        }
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        **options,
    )
    raw_text = (response.choices[0].message.content or "").strip()
    logger.info("千问 API 返回(audit_type=%s): %s", audit_type, raw_text[:500])
    return raw_text


def _judge_items(
    targets: list[RubricItem],
    audit_type: str,
    video_url: str,
    evidence: Optional[VideoEvidence],
    instruction: str,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> list[dict]:
    """只对指定评分项判一次 复用同一份证据

    缺项补问与扣分项投票共用这条路径 保证两处的提示词结构完全一致

    :return: 模型返回的原始 items 数组 编号合法性由调用方校验
    """
    if not targets:
        return []
    target_full = round(sum(item.score for item in targets), 2)
    system_prompt = _build_system_prompt(
        audit_type,
        f"【评分规则】以下 {len(targets)} 个评分项，满分 {target_full:g} 分\n"
        "编号以每项开头的数字为准；每个编号必须单独返回一条，不得用分组名称代替编号，"
        f"也不得把多个评分项合并成一条\n{_format_rubric(targets, group=False)}",
        _RUBRIC_OUTPUT_FORMAT,
        evidence,
    )
    user_content = _build_user_content(
        video_url, evidence, instruction, _prompt_reference_images(targets),
    )
    data = _parse_json(_call_model(system_prompt, user_content, audit_type, temperature))
    patch = data.get("items")
    if not isinstance(patch, list):
        raise ValueError("模型返回缺少 items 数组")
    return [entry for entry in patch if isinstance(entry, dict)]


def score_video(
    video_url: str,
    rule_text: str,
    audit_type: str,
    evidence: Optional[VideoEvidence] = None,
) -> dict:
    """
    按文本规则对视频进行打分

    :param video_url: 视频的 OSS 预签名 URL（无切片证据时由模型直接读整段视频）
    :param rule_text: 打分规则纯文本（从 doc/score_rules/ 加载）
    :param audit_type: 审核类型中文名
    :param evidence: 切片关键帧 + 语音转写证据 为 None 时回退整段视频
    :return: AI 返回的打分结果 dict，包含 total_score / deductions / summary
    """
    system_prompt = _build_system_prompt(
        audit_type,
        f"【评分规则】\n{rule_text}",
        _OUTPUT_FORMAT,
        evidence,
    )
    user_content = _build_user_content(
        video_url,
        evidence,
        "请根据上述评分规则对该操作逐项打分，以 JSON 格式返回结果。",
    )

    logger.info(
        "开始调用千问 API 打分, audit_type=%s, 输入=%s",
        audit_type,
        "切片关键帧+语音" if evidence is not None else "整段视频",
    )
    raw_text = _call_model(system_prompt, user_content, audit_type)
    data = _parse_json(raw_text)
    data["summary"] = _clean_summary(data.get("summary", ""))
    # 文本规则模式仍保留作为兜底 但标记来源: 该口径没有逐项分值 总分由模型自算 服务端无法逐项汇总校验
    data["scoring_mode"] = "legacy_text"
    logger.warning(
        "打分走文本规则模式 audit_type=%s 该类型建议迁移为结构化评分项", audit_type
    )
    return data


def _format_rubric(rubric_items: list[RubricItem], group: bool = True) -> str:
    """将结构化评分项按维度排版为提示词文本

    :param group: 是否输出维度分组标题。只判部分项时必须关闭
                  否则模型会拿维度标题当评分项给分 甚至把同维度几项合并成一条
    """
    lines = []
    current_dimension = None
    for item in rubric_items:
        if group and item.dimension and item.dimension != current_dimension:
            current_dimension = item.dimension
            lines.append(f"\n{current_dimension}")
        lines.append(f"{item.number} {item.point}（{item.score:g}分）")
        if item.evidence:
            lines.append(f"  视频识别依据：{item.evidence}")
        weight = _inline(getattr(item, "oral_visual_weight", ""))
        if weight:
            lines.append(f"  口述视觉权重：{weight}")
        keywords = _inline(getattr(item, "oral_keywords", ""))
        if keywords:
            lines.append(f"  口述关键词：{keywords}")
        remark = _inline(getattr(item, "remark", ""))
        if remark:
            lines.append(f"  备注：{remark}")
    return "\n".join(lines).strip()


def format_rubric_preview(rubric_items: list[RubricItem]) -> str:
    """供规则管理页预览 与写入提示词的评分规则排版保持一致"""
    return _format_rubric(rubric_items)


def score_video_with_rubric(
    video_url: str,
    rubric_items: list[RubricItem],
    audit_type: str,
    evidence: Optional[VideoEvidence] = None,
) -> dict:
    """
    按结构化评分项逐项打分 总分由服务端汇总

    :param video_url: 视频的 OSS 预签名 URL（无切片证据时由模型直接读整段视频）
    :param rubric_items: 从 xlsx 解析出的评分项
    :param audit_type: 审核类型中文名
    :param evidence: 切片关键帧 + 语音转写证据 为 None 时回退整段视频
    :return: 含 full_score / total_score / deductions / items / summary 的 dict
    """
    if not rubric_items:
        raise ValueError("结构化评分项为空")

    full_score = round(sum(item.score for item in rubric_items), 2)
    system_prompt = _build_system_prompt(
        audit_type,
        f"【评分规则】共 {len(rubric_items)} 个评分项，满分 {full_score:g} 分\n{_format_rubric(rubric_items)}",
        _RUBRIC_OUTPUT_FORMAT,
        evidence,
    )
    reference_images = _prompt_reference_images(rubric_items)
    user_content = _build_user_content(
        video_url,
        evidence,
        "请根据上述评分规则逐项打分，以 JSON 格式返回结果。",
        reference_images,
    )

    logger.info(
        "开始调用千问 API 逐项打分, audit_type=%s, items=%d, 规范参考图=%d张, 输入=%s",
        audit_type,
        len(rubric_items),
        len(reference_images),
        "切片关键帧+语音" if evidence is not None else "整段视频",
    )
    raw_text = _call_model(system_prompt, user_content, audit_type)
    return _finalize_rubric_result(raw_text, rubric_items, audit_type, video_url, evidence)


def _assemble_result(
    raw_items: list,
    summary: str,
    rubric_items: list[RubricItem],
) -> tuple[dict, list[str]]:
    """
    校验 AI 逐项结果并汇总总分、扣分项

    :return: (结果 dict, 缺失的评分项编号列表)；缺项不再抛错 由调用方补问或兜底
    """
    # 编号不能只归一化模型侧: 规则里常写成 1/2/3 而模型可能返回 ①/(1)
    # 只归一侧会让全部评分项被判为未知编号丢弃(实测规则编号为 1~9 时 items 全空)
    key_of = _normalize_number
    rubric_by_number = {key_of(item.number): item for item in rubric_items}
    order = {key_of(item.number): idx for idx, item in enumerate(rubric_items)}
    seen: set[str] = set()
    item_results = []
    deductions = []

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            logger.warning("忽略格式错误的评分项: %r", raw_item)
            continue
        number = key_of(raw_item.get("number", ""))
        if number not in rubric_by_number:
            logger.warning("忽略未知评分项编号: %r", raw_item.get("number"))
            continue
        if number in seen:
            logger.warning("忽略重复评分项编号: %s", number)
            continue

        rubric = rubric_by_number[number]
        try:
            score = float(raw_item.get("score"))
        except (TypeError, ValueError):
            # 分数无效视同缺项 交给补问处理
            logger.warning("评分项 %s 的 score 无效: %r", number, raw_item.get("score"))
            continue
        seen.add(number)

        # 越界分数按该项分值区间截断 防止 AI 给出负分或超满分
        score = round(max(0.0, min(float(rubric.score), score)), 2)
        reason = _clean_reason(str(raw_item.get("reason", "")).strip())
        # reason 已经写明各方证据均达成时 分数不能还按份额打折
        score, promoted = _align_score_with_reason(score, reason, float(rubric.score))
        if promoted:
            logger.info(
                "评分项 %s reason 已判定达成 分数由 %.2f 提升至 %.2f",
                rubric.number, float(raw_item.get("score") or 0), score,
            )

        item_results.append({
            "number": rubric.number,        # 统一回规则里的原始编号 与前端、二次补证保持一致
            "dimension": rubric.dimension,
            "point": rubric.point,
            "full_score": rubric.score,
            "score": score,
            "time": str(raw_item.get("time", "")).strip(),   # 证据时刻 供二次补证定位
            "reason": reason,
            "fallback": bool(raw_item.get("fallback")),       # 补问仍缺时的兜底项 需人工复核
        })
        if score < rubric.score:
            deductions.append({
                "item": rubric.label,
                "deduct_score": round(rubric.score - score, 2),
                "reason": reason or "未说明扣分原因",
            })

    missing = [item.number for item in rubric_items if key_of(item.number) not in seen]
    item_results.sort(key=lambda entry: order[key_of(entry["number"])])
    total_score = round(sum(entry["score"] for entry in item_results), 2)

    return {
        "full_score": round(sum(item.score for item in rubric_items), 2),
        "total_score": total_score,
        "deductions": deductions,
        "items": item_results,
        "summary": _clean_summary(summary),
        "scoring_mode": "rubric",
    }, missing


def _reask_missing(
    missing: list[str],
    rubric_items: list[RubricItem],
    audit_type: str,
    video_url: str,
    evidence: Optional[VideoEvidence],
) -> list[dict]:
    """只对缺失评分项补问一次 复用同一份证据 避免整单重跑视频与转写"""
    targets = [item for item in rubric_items if item.number in set(missing)]
    return _judge_items(
        targets, audit_type, video_url, evidence,
        instruction="请只对上述评分项打分，以 JSON 格式返回结果。",
    )


def _finalize_rubric_result(
    raw_text: str,
    rubric_items: list[RubricItem],
    audit_type: str,
    video_url: str,
    evidence: Optional[VideoEvidence],
) -> dict:
    """校验逐项结果: 缺项先补问一次 补问仍缺按证据受限档下限兜底 不让整单失败重跑"""
    data = _parse_json(raw_text)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("AI 返回缺少 items 数组")
    raw_items = [entry for entry in raw_items if isinstance(entry, dict)]
    summary = data.get("summary", "")

    result, missing = _assemble_result(raw_items, summary, rubric_items)
    for _ in range(_MAX_REPAIR_ROUNDS):
        if not missing:
            break
        logger.warning("打分结果缺少评分项 %s 补问一次", missing)
        try:
            raw_items = raw_items + _reask_missing(
                missing, rubric_items, audit_type, video_url, evidence
            )
        except Exception:
            logger.exception("补问缺失评分项失败 直接按兜底处理")
            break
        result, missing = _assemble_result(raw_items, summary, rubric_items)

    if missing:
        logger.error(
            "补问后仍缺少评分项 %s 按证据受限档下限 %.0f%% 兜底",
            missing,
            _MISSING_SCORE_RATIO * 100,
        )
        by_number = {item.number: item for item in rubric_items}
        fallback = [
            {
                "number": number,
                "score": round(float(by_number[number].score) * _MISSING_SCORE_RATIO, 2),
                "reason": _MISSING_ITEM_REASON,
                "fallback": True,
            }
            for number in missing
        ]
        result, _ = _assemble_result(raw_items + fallback, summary, rubric_items)
    return result


# 扣分项投票: 同一份证据换温度重判 逐项取中位数
# 实测同一帧首轮判「无名指悬空」、另一轮判「抵在模型下颌基座形成支点」
# 单次结论不可信 但多次独立判定一致时基本可信 少数服务多数
# 三票时中位数只会取到第二高 两次判错才能纠回一项；五票才能把「两次对一次错」翻过来
# 温度从 0.3 起跳: 首轮已经是 0.1 再从 0.2 起步 五个样本里三个挤在低温区
# 低温轮次实测一致偏严 中位数会被系统性拉低
_VOTE_TEMPERATURES = (0.3, 0.5, 0.7, 0.9)
_VOTE_MAX_ITEMS = 6   # 一次投票最多带几项 项数一多就重演首轮的长上下文漏看
_VOTE_MIN_SAMPLES = 3  # 至少几票才取中位数 两票取中位就是拿首轮和一次重判押宝

# 满分项的 reason 里既没有时刻也没有口述引文 说明模型没拿出证据就给了满分
# 后续投票、补证、仲裁都只往上改 这类虚满分若不在此处复核 就再没有环节能发现
_EVIDENCE_ANCHOR_RE = re.compile(r"\d{1,2}[:：]\d{2}|口述|语音|“|「|\"")


def _lacks_evidence(entry: dict) -> bool:
    """满分项是否缺少可核对的证据: reason 与 time 都没有时刻 也没有引用口述"""
    anchors = f"{entry.get('reason') or ''} {entry.get('time') or ''}"
    return not _EVIDENCE_ANCHOR_RE.search(anchors)


def _vote_temperatures() -> tuple[float, ...]:
    """实际用于投票的温度序列 轮数由 MEDIA_VOTE_ROUNDS 控制"""
    rounds = int(settings.MEDIA_VOTE_ROUNDS or len(_VOTE_TEMPERATURES))
    rounds = max(1, min(rounds, len(_VOTE_TEMPERATURES)))
    return _VOTE_TEMPERATURES[:rounds]


def vote_deducted_items(
    video_url: str,
    rubric_items: list[RubricItem],
    result: dict,
    audit_type: str,
    evidence: Optional[VideoEvidence] = None,
) -> Optional[dict]:
    """只对已扣分与证据空泛的满分项重判若干次 逐项取中位数

    有据可查的满分项不再花调用；证据与首轮完全相同 只换采样温度
    取中位数而不是取最高分: 多数判错才会维持扣分 不把偶发高分当真相
    满分项同理: 多数重判都认为不达标才会扣分 一次偶发低分不动它

    :return: 重算后的结果 dict；无待投项或样本不足时返回 None 由调用方沿用原结果
    """
    entries = {_normalize_number(e["number"]): e for e in result.get("items") or []}
    targets = []
    unproven = []
    for item in rubric_items:
        entry = entries.get(_normalize_number(item.number))
        if entry is None or entry.get("fallback"):
            continue
        if float(entry.get("score") or 0) < float(item.score) - 0.001:
            targets.append(item)
        elif _lacks_evidence(entry) or _contradicts_full_mark(
            float(entry.get("score") or 0), entry.get("reason") or "", float(item.score)
        ):
            unproven.append(item)
    if not targets and not unproven:
        logger.info("扣分项投票: 无扣分项 跳过")
        return None
    if len(targets) > _VOTE_MAX_ITEMS:
        # 扣分越多越可能是误判 也越值得花这几次调用
        targets.sort(key=lambda item: float(entries[_normalize_number(item.number)]["score"]))
        logger.info("扣分项投票: 扣分项 %d 个超限 只投扣分最多的 %d 个", len(targets), _VOTE_MAX_ITEMS)
        targets = targets[:_VOTE_MAX_ITEMS]
    # 虚满分项只占剩余名额 扣分项的误判代价更高
    if unproven:
        room = _VOTE_MAX_ITEMS - len(targets)
        if room > 0:
            logger.info(
                "扣分项投票: 满分项 %s 缺少证据或结论自相矛盾 一并复核",
                [item.number for item in unproven[:room]],
            )
            targets += unproven[:room]

    # 每个编号先放首轮那一票
    votes: dict[str, list[dict]] = {
        _normalize_number(item.number): [entries[_normalize_number(item.number)]]
        for item in targets
    }
    for temperature in _vote_temperatures():
        try:
            patch = _judge_items(
                targets, audit_type, video_url, evidence,
                "请独立判定上述评分项，以 JSON 格式返回结果。",
                temperature,
            )
        except Exception:
            logger.exception("扣分项投票第 %.1f 轮失败 该轮不计入", temperature)
            continue
        for entry in patch:
            number = _normalize_number(entry.get("number", ""))
            if number in votes:
                votes[number].append(entry)

    expected = 1 + len(_vote_temperatures())
    changes = []
    for item in targets:
        number = _normalize_number(item.number)
        # 满分票的 reason 却写着没做到 这一票到底投的哪边说不清 不计入
        pool = [
            entry for entry in votes[number]
            if not _contradicts_full_mark(
                float(entry.get("score") or 0),
                _clean_reason(str(entry.get("reason") or "")),
                float(item.score),
            )
        ]
        # 票数为偶数时去掉首轮那票: 该项正是因首轮存疑才来投票 奇数票的中位数才是真实发生过的判定
        # 样本不齐时不拿两个值凑平均 那会造出 0.18 这类并不存在的档位
        if len(pool) % 2 == 0:
            pool = pool[1:]
        if len(pool) < _VOTE_MIN_SAMPLES:
            logger.warning(
                "扣分项 %s 投票样本不足(%d/%d) 维持首轮", item.number, len(pool), expected
            )
            continue
        # 每一票先按它的 reason 对齐再排序 否则结论说达标、分数填打折值的票会被当成低分票
        def _aligned_vote(entry: dict) -> float:
            score, _ = _align_score_with_reason(
                float(entry.get("score") or 0),
                _clean_reason(str(entry.get("reason") or "")),
                float(item.score),
            )
            return score

        ordered = sorted(pool, key=_aligned_vote)
        middle = ordered[len(ordered) // 2]
        old = float(entries[number].get("score") or 0)
        new = round(max(0.0, min(float(item.score), _aligned_vote(middle))), 2)
        if abs(new - old) < 0.01:
            continue
        entries[number]["score"] = new
        entries[number]["reason"] = _clean_reason(str(middle.get("reason") or ""))
        entries[number]["time"] = str(middle.get("time") or "").strip()
        changes.append({"number": item.number, "point": item.point, "from": old, "to": new})

    if not changes:
        logger.info("扣分项投票: %d 项参与 结论与首轮一致", len(targets))
        return None

    rebuilt = _rebuild_from_entries(result, rubric_items)
    rebuilt["vote"] = {"items": len(targets), "changed": changes}
    logger.info("扣分项投票: %d 项参与 调整 %d 项", len(targets), len(changes))
    return rebuilt


def _rebuild_from_entries(result: dict, rubric_items: list[RubricItem]) -> dict:
    """分数已在 entries 上改过时  复用首轮那套汇总重算总分与扣分项

    避免投票与冲突仲裁各自写一份汇总算法与首轮分叉
    """
    raw = [
        {
            "number": entry["number"],
            "score": entry["score"],
            "reason": entry.get("reason", ""),
            "time": entry.get("time", ""),
            "fallback": entry.get("fallback", False),
        }
        for entry in result.get("items") or []
    ]
    rebuilt, _ = _assemble_result(raw, result.get("summary", ""), rubric_items)
    return rebuilt


_CONFLICT_POLICY = """【冲突检查说明】
1. 下面各项判定来自同一段视频、同一份关键帧与同一份语音转写。
2. 你的任务不是重新打分，只检查它们之间有没有对同一客观事实给出相反认定。
3. 典型冲突：一项写「画面可见某器械被使用」，另一项写「全程未见该器械」「该器械未使用」；一项写「左手持口镜」，另一项写「左手未持口镜」「未见口镜」。两项评分点不同并不影响：只要它们对同一件器械、同一只手、同一个动作是否发生给出相反认定，就是冲突。
4. 不得发明新的判定，不得改分，也不得把措辞与程度差异报成冲突。"""

_CONFLICT_FORMAT = """请严格按照以下 JSON 格式返回，不要输出任何多余内容：
{
  "conflicts": [
    {
      "fact": "<被矛盾的单一客观事实，一句话，如「左手是否持口镜」>",
      "numbers": ["<参与冲突的评分项编号>", "<编号>"]
    }
  ]
}
要求：
1. 只报同一件客观事实（某器械是否存在或被使用、某只手是否持物、某部位是否被接触到、某动作是否发生）在一项被认定发生、在另一项被认定从未发生的冲突。
2. 程度差异不算冲突（如「支点稳定」与「支点略松」），说的是不同器械或不同部位也不算（如一项说探针、一项说口镜）；但评分点不同而对同一器械、同一只手给出相反认定的必须报。
3. 每组冲突给 2 到 3 个编号，同一组必须是互相排斥的那几项。
4. 没有冲突时 conflicts 返回空数组。"""

_CONFLICT_MAX_GROUPS = 2   # 最多仲裁几组 每组一次带证据调用


def _detect_conflicts(result: dict, audit_type: str) -> list[dict]:
    """纯文本问一次: 各项判定之间有没有对同一事实的相反认定

    不带视频与图片 只把各项 reason 交给模型做文本比对 成本接近于零
    :return: [{"fact": 描述, "numbers": {归一化编号}}]
    """
    lines = []
    for item in result.get("items") or []:
        if item.get("fallback"):
            continue
        lines.append(
            f"{item['number']} {item['point']}"
            f"（得{float(item['score']):g}/{float(item['full_score']):g}分）："
            f"{item.get('reason') or '未说明'}；证据时刻：{item.get('time') or '无'}"
        )
    if len(lines) < 2:
        return []
    system_prompt = "\n".join([
        "你是考核评分结果的质检员，只负责找出同一份证据下各项判定之间的矛盾，不重新打分。",
        "",
        _CONFLICT_POLICY,
        "",
        "【输出要求】",
        _CONFLICT_FORMAT,
    ])
    user_content = [{"type": "text", "text": "【各项判定】\n" + "\n".join(lines)}]
    data = _parse_json(_call_model(system_prompt, user_content, audit_type))
    groups = data.get("conflicts")
    if not isinstance(groups, list):
        return []
    found = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        numbers = {_normalize_number(n) for n in (group.get("numbers") or [])}
        if len(numbers) < 2:
            continue
        found.append({"fact": str(group.get("fact") or "").strip(), "numbers": numbers})
    return found[:_CONFLICT_MAX_GROUPS]


def arbitrate_conflicts(
    video_url: str,
    rubric_items: list[RubricItem],
    result: dict,
    audit_type: str,
    evidence: Optional[VideoEvidence] = None,
) -> Optional[dict]:
    """把结果里对同一事实给出相反认定的评分项打包重判一次

    补证是链路最后一环 它改完分数后没人回头检查其余项
    实测同一段证据下第 4 项写「左手持口镜」 第 2 项同时入库的却是「全程未见口镜」
    这类矛盾不可能同时为真 拿同一份证据把冲突项放同一次调用里重判 迫使模型择一

    :return: 重算后的结果 dict；无冲突或无改动时返回 None 由调用方沿用原结果
    """
    groups = _detect_conflicts(result, audit_type)
    if not groups:
        logger.info("冲突仲裁: 未发现互相矛盾的判定")
        return None

    entries = {_normalize_number(e["number"]): e for e in result.get("items") or []}
    by_number = {_normalize_number(item.number): item for item in rubric_items}
    changes: list[dict] = []
    facts: list[str] = []
    for group in groups:
        targets = [by_number[n] for n in group["numbers"] if n in by_number and n in entries]
        if len(targets) < 2:
            logger.warning("冲突仲裁: 编号对不上评分项 跳过 %s", sorted(group["numbers"]))
            continue
        instruction = (
            f"以下评分项对同一事实「{group['fact'] or '见各项描述'}」给出了相反的认定。"
            "请先只依据证据确定这一事实到底成立与否 再按该结论给这几项打分 各项之间不得再自相矛盾。"
        )
        try:
            patch = _judge_items(targets, audit_type, video_url, evidence, instruction)
        except Exception:
            logger.exception("冲突仲裁调用失败 维持原判定: %s", sorted(group["numbers"]))
            continue
        fresh = {_normalize_number(entry.get("number", "")): entry for entry in patch}
        touched = False
        for item in targets:
            number = _normalize_number(item.number)
            entry = fresh.get(number)
            if entry is None:
                continue
            reason = _clean_reason(str(entry.get("reason") or ""))
            bounded = round(max(0.0, min(float(item.score), float(entry.get("score") or 0))), 2)
            score, _ = _align_score_with_reason(bounded, reason, float(item.score))
            old = float(entries[number].get("score") or 0)
            # 仲裁只负责把被判 0 的那一项抬上来 不能反过来把已给分的项压回去
            # 实测补证用窗口内 1 秒密帧认出左手持口镜给到 0.15 仲裁拿首轮稀疏关键帧重判又写成「未见」
            # 首轮证据比补证弱 弱证据不得推翻强结论；判 0 的代价最高 宁可保留可解释的小矛盾
            if score < old and old > 0.01:
                logger.info(
                    "冲突仲裁: 评分项 %s 已有 %.2f 分 不因重判降级", item.number, old
                )
                continue
            if abs(score - old) < 0.01:
                continue
            # 抬分必须有站得住的依据: 没写理由或理由与满分相反时 维持原判定
            if score > old and (not reason or _contradicts_full_mark(score, reason, float(item.score))):
                logger.info("冲突仲裁: 评分项 %s 改判理由缺失或与分数矛盾 维持原判定", item.number)
                continue
            entries[number]["score"] = score
            entries[number]["reason"] = reason
            entries[number]["time"] = str(entry.get("time") or "").strip()
            changes.append({
                "number": item.number, "point": item.point,
                "from": old, "to": score, "fact": group["fact"],
            })
            touched = True
        if touched:
            facts.append(group["fact"])

    if not changes:
        logger.info("冲突仲裁: 检出 %d 组 重判后结论自洽 无需改动", len(groups))
        return None

    rebuilt = _rebuild_from_entries(result, rubric_items)
    rebuilt["conflicts"] = {"groups": len(groups), "facts": facts, "changed": changes}
    logger.info("冲突仲裁: 检出 %d 组 调整 %d 项", len(groups), len(changes))
    return rebuilt


# 二次补证: 只看连续片段复核存疑评分项
_SUPPLEMENT_FORMAT = """请严格按照以下 JSON 格式返回复核结果，不要输出任何多余内容：
{
  "items": [
    {
      "number": "<评分项编号>",
      "clear": <true/false, 片段中能否清楚看到该项要求的关键动作>,
      "score": <float, 复核后的该项得分；clear 为 false 时填 0，口述例外见要求 2>,
      "reason": "<判定依据，clear 为 true 时必须引用片段中的具体画面>"
    }
  ]
}
要求：
1. 只输出【复核评分项】中列出的编号，每个恰好一次，不得增删。
2. clear 为 false 时服务端一律保留上一轮分数，score 填 0 即可，不得借“看不清”加减分；唯一例外是本轮片段语音已把该项动作要素说清且评分尺度第 2 条第 1 档的前提成立，此时 score 填该项满分。
3. score 范围为 0 到该项分值，判定尺度与第一轮完全一致。
4. reason 用陈述句，不超过 100 字，不得出现算式、分值换算，也不得写思考过程（反问、改口、再核算），只写一次最终结论。"""

_SUPPLEMENT_POLICY = """【二次复核说明】本次输入是上一轮按关键帧判定后存在证据缺口的评分项，以及按时间切出的连续视频片段（动作过程连续可见）。
1. 先判断 clear：片段里能否看清该项要求的关键动作细节（器械、部位、手法、次数、时长、是否污染或跨越无菌区）。
2. clear 为 true 时按评分尺度重新给分，reason 必须写出片段中看到的具体画面，不得笼统写“已完成”“规范”。
3. clear 为 false 时默认保留上一轮分数，reason 说明无法确认的原因；片段没拍到不等于做得不对，也不等于做到了。例外：本轮片段语音把该项动作要素说清、而画面未呈现时，若评分尺度第 2 条第 1 档的前提成立（该项未填口述视觉权重或权重全部归口述），按该档给满分，reason 引用口述原文——口述完整与 clear 无关。
4. 只依据随附片段与语音转写判定，不得引入第一轮之外的其他推测。
5. 语音转写标注「语音转写失败」时，表示该段口述没有被转写出来，不代表考生没说话，不得据此认定口述未交代。"""


def _as_bool(value) -> bool:
    """宽松解析布尔值 兼容模型返回的字符串 true/false"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "是")
    return bool(value)


def rescore_with_clips(
    audit_type: str,
    check_items: list[dict],
    clips: list[dict],
    blind: bool = False,
) -> list[dict]:
    """
    二次复核: 用连续视频片段重判证据受限的评分项

    :param audit_type: 审核类型中文名
    :param check_items: 待复核评分项 每项含 number/point/evidence/full_score/score/reason/time
    :param clips: 片段列表 每项含 label/url/transcript
    :param blind: 不告诉模型上一轮结论 避免它顺着旧判定找理由 用于判 0 分的逐项复核
    :return: [{number, clear, score, reason}] 仅含成功解析且编号在复核范围内的项
    """
    if not check_items or not clips:
        return []

    lines = [f"【复核评分项】共 {len(check_items)} 项"]
    for item in check_items:
        if blind:
            # 上一轮分数也是结论: 写着「上一轮0分」等于把旧判定又递了回去
            lines.append(f"{item['number']} {item['point']}（满分{item['full_score']:g}分）")
        else:
            lines.append(
                f"{item['number']} {item['point']}（满分{item['full_score']:g}分，上一轮{item['score']:g}分）"
            )
        if item.get("evidence"):
            lines.append(f"  视频识别依据：{item['evidence']}")
        if item.get("weight"):
            lines.append(f"  口述视觉权重：{_inline(item['weight'])}")
        if item.get("keywords"):
            lines.append(f"  口述关键词：{_inline(item['keywords'])}")
        if item.get("remark"):
            lines.append(f"  备注：{_inline(item['remark'])}")
        if blind:
            lines.append("  上一轮结论与分数均不予提供：请完全依据下方片段与关键帧独立判定")
        else:
            lines.append(f"  上一轮判定：{item.get('reason') or '无'}；证据时刻：{item.get('time') or '无'}")

    system_prompt = "\n".join([
        "你是一位严格、客观的口腔医学操作考核评分专家，正在对上一轮存疑的评分项做二次复核。",
        _scoring_policy("\n".join(lines)),
        "",
        _SUPPLEMENT_POLICY,
        "",
        "【输出要求】",
        _SUPPLEMENT_FORMAT,
    ])

    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for clip in clips:
        content.append({
            "type": "text",
            "text": f"【片段{clip['label']}】语音转写：{clip.get('transcript') or '（该片段无语音内容）'}",
        })
        content.append({"type": "video_url", "video_url": {"url": clip["url"]}})
        frames = clip.get("frames") or []
        if frames:
            content.append({
                "type": "text",
                "text": (
                    f"片段{clip['label']} 的静态关键帧。@mm:ss 是该帧在本段视频里的准确时刻，"
                    "分辨率高于视频采样帧：判断器械形态与型号、持械分工、镜面朝向、支点位置、"
                    "握持姿势这类细节时，以这些静态帧为准。"
                ),
            })
            for frame_label, frame_b64 in frames:
                content.append({"type": "text", "text": f"关键帧 @{frame_label}："})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                })
    content.append({"type": "text", "text": "请对上述评分项逐项复核，以 JSON 格式返回结果。"})

    raw_text = _call_model(system_prompt, content, audit_type)
    return _parse_supplement(raw_text, check_items)


def _parse_supplement(raw_text: str, check_items: list[dict]) -> list[dict]:
    """校验二次复核返回: 忽略未知编号与无法解析的项 分数按该项分值截断"""
    data = _parse_json(raw_text)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("二次复核返回缺少 items 数组")

    pending = {_normalize_number(item["number"]): item for item in check_items}
    results = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        number = _normalize_number(raw.get("number", ""))
        item = pending.pop(number, None)
        if item is None:
            continue
        try:
            score = float(raw.get("score"))
        except (TypeError, ValueError):
            continue
        full = float(item["full_score"])
        clean = _clean_reason(str(raw.get("reason", "")).strip())
        bounded = round(max(0.0, min(full, score)), 2)
        clear = _as_bool(raw.get("clear"))
        # 复核结论说达成时 同样不能让分数停在打折值 只有 clear=false 保留原分时不计
        if clear:
            bounded, _ = _align_score_with_reason(bounded, clean, full)
        # clear=true 必须写出片段中的具体画面 没写或结论与满分相反时这次改判不可信 按看不清处理
        # 否则分数改了 reason 却留着上一轮「未执行」 冲突检测读到的仍是旧结论
        if clear and (not clean or _contradicts_full_mark(bounded, clean, full)):
            logger.info("二次复核: 评分项 %s 改判理由缺失或与分数矛盾 按看不清处理", item["number"])
            clear = False
            bounded = 0.0
        results.append({
            "number": item["number"],      # 回填规则原始编号 供 _apply_updates 对上结果项
            "clear": clear,
            "score": bounded,
            "reason": clean,
        })
    return results


def _parse_json(text: str) -> dict:
    """
    从 AI 返回文本中解析 JSON，兼容 markdown 代码块包裹

    :param text: AI 返回的原始文本
    :return: 解析后的 dict
    :raises ValueError: 无法解析时抛出
    """
    # 去掉 markdown 代码块标记
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 去掉首尾的 ``` 行
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试从文本中提取第一个 JSON 对象
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 AI 返回中解析 JSON: {text[:200]}")
