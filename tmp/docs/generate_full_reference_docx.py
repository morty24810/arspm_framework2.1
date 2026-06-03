# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import re
import shutil
from xml.etree import ElementTree as ET


ROOT = Path("/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1")
SRC = Path("/Users/Morty/Desktop/毕业设计/本科毕业设计 傅懋杰.docx")
OUT = ROOT / "output/doc/本科毕业设计 傅懋杰_第一章扩充参考文献完整版.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
ET.register_namespace("w", W_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture")


def q(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS)).strip()


def direct_paragraphs(body: ET.Element):
    return [child for child in list(body) if child.tag == q("p")]


def direct_child_index_by_text(body: ET.Element, text: str) -> int:
    for idx, child in enumerate(list(body)):
        if child.tag == q("p") and paragraph_text(child) == text:
            return idx
    raise ValueError(f"Paragraph not found: {text}")


def direct_paragraph_by_contains(body: ET.Element, needle: str) -> ET.Element:
    for child in list(body):
        if child.tag == q("p") and needle in paragraph_text(child):
            return child
    raise ValueError(f"Paragraph containing text not found: {needle}")


def first_run_rpr(template: ET.Element):
    run = template.find("w:r", NS)
    if run is None:
        return None
    rpr = run.find("w:rPr", NS)
    return deepcopy(rpr) if rpr is not None else None


def add_text_run(p: ET.Element, text: str, base_rpr=None, superscript=False):
    if not text:
        return
    r = ET.SubElement(p, q("r"))
    rpr = deepcopy(base_rpr) if base_rpr is not None else ET.Element(q("rPr"))
    if superscript:
        for old in list(rpr.findall("w:vertAlign", NS)):
            rpr.remove(old)
        vert = ET.SubElement(rpr, q("vertAlign"))
        vert.set(q("val"), "superscript")
    if len(list(rpr)) or rpr.attrib:
        r.append(rpr)
    t = ET.SubElement(r, q("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text


CITE_RE = re.compile(r"(\[\d+\])")


def make_paragraph(text: str, template: ET.Element, superscript_cites=True) -> ET.Element:
    p = ET.Element(q("p"))
    ppr = template.find("w:pPr", NS)
    if ppr is not None:
        p.append(deepcopy(ppr))
    base_rpr = first_run_rpr(template)
    if superscript_cites:
        for part in CITE_RE.split(text):
            if not part:
                continue
            add_text_run(p, part, base_rpr, superscript=bool(CITE_RE.fullmatch(part)))
    else:
        add_text_run(p, text, base_rpr, superscript=False)
    return p


def replace_between(body: ET.Element, start_text: str, end_text: str, new_paras):
    children = list(body)
    start = direct_child_index_by_text(body, start_text)
    end = direct_child_index_by_text(body, end_text)
    if not start < end:
        raise ValueError(f"Invalid replacement bounds: {start_text} -> {end_text}")
    body[:] = children[: start + 1] + list(new_paras) + children[end:]


intro_paras = [
    "随着智能制造与工业4.0的发展，制造系统正在由传统的静态生产组织方式转向数据驱动、互联协同和自主决策的智能系统形态。工业4.0强调通过信息物理系统、工业物联网与数据分析技术实现生产要素的实时感知、互联互通和柔性优化，其核心目标是在复杂多变的生产环境中提高资源利用率、交付可靠性与系统韧性[1]。在此背景下，设备状态不再只是生产过程的外部约束，而是直接影响加工能力、维护时机和调度决策的重要系统变量。",
    "柔性作业车间调度问题是离散制造系统中的典型优化问题。与传统作业车间相比，柔性作业车间不仅需要确定工序加工顺序，还需要在可行机器集合中选择具体加工设备；当系统进一步存在动态作业到达、机器健康状态变化和维护活动插入时，调度决策会表现出更强的实时性和耦合性。已有研究已经将深度强化学习用于动态柔性作业车间调度、新作业插入、多目标调度以及多智能体制造系统调度等问题[2][3][4][5]，说明学习型调度方法能够在复杂状态空间中形成自适应决策能力。",
    "另一方面，预测性维护通过对设备运行数据进行建模，估计剩余使用寿命并提前规划维护活动，可以降低突发故障对生产系统造成的停机风险。近年来，已有研究尝试将RUL预测结果与实时调度、维护决策和多智能体强化学习结合，用于生产与维护的协同优化[6]；同时，预测性维护在工业4.0中的跨行业应用也表明，维护决策正在由固定周期或故障后维修逐步转向基于状态和寿命预测的主动维护[7]。因此，如何在同一决策框架下协调设备退化、维护选择与生产调度，是本文研究的主要问题来源。",
    "本文围绕设备剩余使用寿命预测、维护决策与动态柔性作业车间调度之间的耦合关系展开研究。方法设计上，本文以强化学习理论为决策建模基础[8]，借鉴深度Q网络、近端策略优化和蒙特卡洛规划等方法在序贯决策中的思想[9][10][11]；数据建模上，本文采用公开工业退化数据构造RUL预测任务，并结合实际实验数据集完成模型训练与验证[12][13]。本文第一章将从研究背景、国内外研究现状与论文结构三个方面展开论述，为后续问题建模、算法设计和实验分析奠定基础。",
]

background_paras = [
    "从技术发展脉络看，制造系统智能化经历了从自动化、数字化到网络化、智能化的持续演进。工业4.0相关研究指出，现代制造系统的关键特征在于横向集成、纵向集成和端到端工程集成，生产现场的设备、作业、物流和信息系统需要在统一架构下形成闭环反馈[14]。信息物理系统进一步将物理设备状态、传感器数据、网络通信和决策模型联结起来，使生产系统能够依据实时数据进行状态感知和自适应控制[15]。预测性制造的提出则强调通过设备健康评估、故障诊断和性能预测，在故障发生前调整生产计划与维护策略[16]。这些研究共同说明，设备健康状态正在成为制造系统运行决策中不可忽略的核心变量。",
    "在传统制造环境中，设备维护通常采用事后维修或定期预防性维护。事后维修虽然策略简单，但故障发生具有随机性，容易导致生产中断、交期延误和成本上升；定期预防性维护能够降低故障风险，但维护周期往往依赖经验设定，可能出现过度维护或维护不足。随着传感器、工业数据库和机器学习技术的发展，基于状态的维护与预测性维护逐渐成为研究重点[17]。基于状态的维护利用设备运行特征判断当前健康状态，预测性维护则进一步估计未来退化趋势和剩余使用寿命，从而为维护窗口选择、备件准备和生产排程提供依据[18]。",
    "预测性维护的核心在于利用历史退化数据和当前观测信息推断设备未来可用性。现有综述表明，预测性维护研究主要包括数据采集、特征构造、健康状态识别、RUL预测和维护策略优化等环节[19][20]。其中，RUL预测直接刻画设备从当前状态到失效阈值之间的剩余时间或剩余周期，是维护决策和生产调度耦合的关键接口。若RUL估计偏大，设备可能在加工过程中发生故障；若RUL估计偏小，系统可能过早安排维护并牺牲产能。因此，RUL预测不仅是单独的机器学习问题，也是后续调度与维护决策的状态输入来源。",
    "RUL建模方法大体可以分为基于物理机理的方法、统计随机过程方法和数据驱动方法。基于物理机理的方法依赖设备退化机理和失效模型，具有较强可解释性，但在复杂装备和多工况环境下建模成本较高；统计方法通过随机过程或概率模型描述退化不确定性，适合对寿命分布和置信区间进行分析；数据驱动方法则从历史运行数据中学习退化特征与RUL之间的非线性映射，更适合传感器数据丰富但机理模型难以精确建立的工业场景[21][22]。机械健康预测研究进一步指出，多源信号融合、时序依赖建模和跨工况泛化能力是提升RUL预测效果的重要方向[23]。",
    "深度学习的发展为RUL预测提供了更强的序列特征提取能力。长短期记忆网络能够通过门控结构缓解长序列训练中的梯度消失问题[24]，门控循环单元在保留序列记忆能力的同时简化了门控结构[25]，相关实验比较表明GRU在若干序列建模任务中能够以较少参数取得接近或优于LSTM的表现[26]。在RUL预测领域，循环神经网络、LSTM、卷积神经网络以及面向制造系统的深度序列模型均被用于从多维传感序列中提取退化模式[27][28][29][30]。因此，本文采用GRU对设备退化序列进行建模，符合工业时序预测中对动态依赖和模型复杂度平衡的需求。",
    "与设备健康预测相对应，生产调度研究关注有限资源下的工序排序和资源分配问题。经典调度理论通常在确定性作业、确定加工时间和静态机器可用性的假设下建立模型，通过目标函数优化完工时间、拖期、机器负载或综合成本[31]。在实际生产中，调度规则是一类重要的快速决策方法，常见规则包括最短加工时间、最早交期、剩余松弛时间和关键比率等[32]。这些规则计算简单、实时性好，但通常只依据局部信息进行贪心选择，难以同时处理动态到达、机器柔性、健康约束与维护活动插入带来的全局影响[33]。",
    "柔性作业车间调度问题进一步放宽了传统作业车间中工序与机器一一对应的约束，允许同一工序在多个候选机器上加工。早期研究从多用途机器调度、禁忌搜索、局部搜索和多目标优化等角度对该问题进行了建模与求解[34][35][36][37]，后续研究又引入遗传算法、变邻域下降和混合启发式方法以提高求解效率和解质量[38]。相关综述表明，柔性作业车间调度具有组合空间大、机器选择与工序排序相互耦合、约束条件多样等特点，是典型的NP-hard问题[39][40]。当作业动态到达或机器状态实时变化时，静态优化结果容易失效，系统需要具备持续重调度能力[41][42]。",
    "生产调度与维护计划的集成优化是本文研究背景中的另一条重要发展线索。传统研究通常将维护视为机器可用时间的外部限制，先确定维护计划再进行生产排程，或在生产计划完成后再插入维护活动。这种分阶段处理方式忽略了维护活动对交期、机器负载和后续工序可行性的影响。已有研究从预防性维护与生产调度联合优化、维护窗口选择、机器组维护和生产计划协同等角度展开探索[43][44][45][46]。这些研究说明，维护活动本质上会占用机器时间并改变未来机器可用性，必须与生产调度共同建模。",
    "在动态柔性作业车间中，作业并不一定在初始时刻全部可见，而是可能随时间陆续进入系统。每个作业通常包含多道具有先后约束的工序，不同工序又具有不同的候选机器集合和机器相关加工时间。与此同时，作业交期、权重、交期紧度系数以及当前系统负载会共同决定调度决策的紧迫性。若只依据单个工序加工时间进行排序，可能导致高权重或交期紧迫作业被延后；若只依据交期排序，又可能造成短作业长时间等待并降低机器利用率。因此，动态调度问题需要同时平衡局部加工效率、全局交付风险和未来机器可用性。",
    "设备健康约束使上述问题进一步复杂化。健康状态较差的机器虽然在当前时刻可能仍可加工，但其RUL可能不足以覆盖长工序或连续加工负荷；如果调度策略忽视这一点，就可能将关键工序分配给高风险设备，进而触发故障停机或额外维护。相反，如果系统过度保守，将低RUL设备长期闲置或频繁安排维护，又会造成产能浪费。由此可见，RUL预测在生产调度中的作用不是简单地判断设备是否失效，而是为工序-机器匹配提供健康风险尺度，使调度智能体能够在加工效率与可靠性之间进行权衡。",
    "综上，本文面对的是一个同时包含设备退化感知、RUL预测、维护选择和柔性调度的动态决策问题。该问题的难点不只在于单一算法的精度，而在于多个子系统之间存在反馈关系：RUL预测影响维护智能体对维护时机和维护类型的判断，维护决策改变机器可用性和健康状态，调度智能体又需要在交期、加工时间、机器负载和健康风险之间进行权衡。因此，本文研究的背景可以概括为：在智能制造环境下，生产系统需要从“只考虑工序与机器匹配”的传统调度，转向“同时考虑设备健康、维护活动与动态交付压力”的协同决策。",
]

status_paras = [
    "现有研究可以从RUL预测、维护决策、柔性作业车间调度以及生产维护协同四个方面进行归纳。首先，在RUL预测方面，研究重点已经由依赖人工特征和浅层回归模型逐渐转向深度时序模型。循环神经网络能够自然处理时间序列数据，适合描述设备退化过程中前后观测之间的相关性；LSTM和GRU通过门控机制保留历史信息，能够缓解普通循环神经网络难以学习长期依赖的问题[24][25][26]。在工业退化数据中，传感器特征往往具有噪声、多维、非线性和工况差异，深度模型可以通过端到端训练学习隐含退化表示，因此在RUL预测任务中被广泛采用[27][28][29][30]。",
    "但是，单纯提高RUL预测精度并不足以保证生产系统性能提升。原因在于，维护决策并不是只由当前RUL数值决定，还受到当前作业队列、机器负载、交期压力和维护成本的影响。例如，当某台设备RUL较低但当前没有关键工序等待时，立即维护可能是合理选择；反之，当设备RUL仍能覆盖短工序加工且作业交期紧迫时，推迟维护可能更有利于系统目标。因此，RUL预测在本文中被定位为维护与调度决策的状态变量，而不是独立优化目标。这样的定位也与预测性维护研究中“由健康评估支持运维决策”的思想一致[18][19][20]。",
    "在生产调度方面，传统规则方法具有解释性强和计算速度快的优势。最短加工时间规则倾向于快速释放作业，最早交期规则强调交付风险控制，松弛时间规则考虑剩余加工负荷与交期之间的差距，关键比率规则则通过剩余时间与剩余加工时间的比值判断作业紧迫程度[32][33]。这些规则在论文实验中可作为对比基准，用于衡量学习型调度智能体是否能够在复杂状态下超越固定启发式策略。然而，规则方法通常难以利用长期回报信息，当维护活动改变未来机器可用性时，局部最优选择可能导致后续拥塞和拖期累积。",
    "元启发式方法是柔性作业车间调度中的另一类重要方法。禁忌搜索、遗传算法、局部搜索和变邻域下降等方法能够在较大的组合空间中搜索高质量解，并在静态或准静态问题中取得较好效果[35][36][38]。但是，元启发式方法通常需要较多迭代计算，当系统频繁出现新作业到达、机器维护插入或设备健康状态变化时，完全重新优化的计算成本较高。同时，本文研究的问题具有事件驱动特征，智能体需要在每个决策时刻快速给出可执行动作，因此本文更关注能够在线决策的强化学习和规划方法。",
    "在分层决策方面，时间抽象和选项框架表明，复杂序贯决策可以拆分为不同层级的子策略或宏动作，从而降低单一决策器同时处理全部状态和动作的难度[47]。本文中的维护决策与调度决策具有天然层次结构：维护智能体更关注机器健康、RUL和维护收益，调度智能体更关注可调度工序、候选机器和交期表现。将二者分开建模可以减少动作空间耦合，并使维护策略与生产调度策略在统一环境反馈下协同演化。",
    "强化学习为动态调度问题提供了以状态、动作和奖励为核心的序贯决策建模框架。Q-learning证明了在有限马尔可夫决策过程中通过采样更新动作价值函数的可行性[48]，深度Q网络进一步利用神经网络近似高维状态下的动作价值函数[9]，Double DQN通过降低价值估计偏差改善了DQN的稳定性[49]。在连续或较大动作空间中，PPO通过限制新旧策略更新幅度提升策略梯度训练稳定性[10]。这些方法为本文维护智能体和调度智能体的设计提供了理论基础。",
    "对于部分可观测或未来不确定性较强的问题，蒙特卡洛规划方法能够通过采样模拟评估候选动作的长期效果。POMCP将蒙特卡洛树搜索与粒子信念表示结合，可在较大POMDP中进行在线规划[11]。在本文调度场景中，未来作业到达、维护效果和机器状态变化会带来不确定性，因此POMCP类方法可作为基于模拟搜索的调度策略对比，用于分析学习型策略与在线规划策略之间的差异。",
    "从已有研究局限看，当前工作仍存在三个不足。第一，部分调度研究默认机器始终可用，未显式考虑设备健康退化和维护占机对调度可行性的影响；第二，部分预测性维护研究只关注RUL预测误差，缺少与生产交期、机器负载和作业权重等调度指标的联动分析；第三，生产维护联合优化研究中常见的静态模型难以直接适用于动态作业到达和事件驱动决策环境。因此，本文需要在问题建模中同时刻画工序顺序约束、机器候选约束、设备健康约束和维护占用约束，并在算法设计中分别构造RUL预测模块、维护智能体和调度智能体。",
    "与现有工作的另一个区别在于，本文不将RUL预测与调度优化割裂处理。RUL模型的输出会进入维护智能体和调度智能体的状态描述，维护动作又会改变后续设备健康状态和机器可用时间，从而影响调度环境的转移过程。这样的闭环结构要求算法既能利用当前观测，又能考虑当前动作对未来状态的影响。对于维护智能体而言，过早维护会带来不必要的维护成本和产能损失，过晚维护会增加故障风险；对于调度智能体而言，选择短期加工时间最小的机器不一定能带来长期最优结果，因为该机器未来可能因健康下降而不可用。",
    "本文在实验设计中保留规则调度、基于价值函数的调度、基于策略优化的调度以及在线规划调度等不同类型的对比方法，其目的并不是简单比较算法名称，而是检验不同决策机制在同一动态制造环境下的表现差异。规则方法体现了传统启发式调度的局部决策能力，DQN类方法体现了离散动作价值学习能力，PPO类方法体现了参数化策略直接优化能力，POMCP类方法则体现了基于模拟搜索的在线规划能力。通过这些对比，可以更清楚地分析设备健康信息、维护决策和调度策略之间的作用机制。",
    "基于上述分析，本文的研究定位不是单独提出一个RUL预测模型，也不是只解决标准柔性作业车间调度问题，而是构建一个面向设备健康感知的生产维护协同决策框架。该框架以GRU模型提供设备RUL估计，以维护智能体判断维护动作，以调度智能体完成工序与机器匹配，并通过仿真环境统一评价交期、完工时间、维护成本和故障风险等指标。这样的研究思路能够更贴近智能制造系统中“预测-决策-执行-反馈”的闭环运行逻辑。",
]

structure_paras = [
    "本文围绕设备退化感知、维护决策与动态柔性作业车间调度的协同优化展开研究，全文结构安排如下。",
    "第一章为绪论。该章首先介绍智能制造、预测性维护和柔性作业车间调度的发展背景，分析设备健康状态进入生产调度决策后带来的新问题；随后从RUL预测、调度规则、强化学习调度和生产维护协同等方面梳理国内外研究现状，归纳现有研究的不足；最后给出本文的研究内容和章节结构。",
    "第二章介绍基础理论与背景知识。该章主要说明柔性作业车间调度、设备退化建模、RUL预测、强化学习、深度强化学习以及蒙特卡洛规划等基础概念，为后续建立数学模型和设计算法提供理论支撑。",
    "第三章进行问题描述与数学建模。该章将生产系统描述为考虑动态作业到达、柔性机器分配、设备健康约束和维护活动耦合的事件驱动柔性作业车间调度问题，并分别给出设备退化与RUL预测建模、维护决策建模、调度动作空间、约束条件和目标函数。",
    "第四章进行算法设计。该章依据第三章模型，将整体方法划分为RUL预测模块、维护智能体和调度智能体三个部分，分别说明GRU预测模型、THDQN或PPO维护策略、DQN、PPO与POMCP调度策略的状态表示、动作定义、奖励设计和决策流程，并给出相应伪代码。",
    "第五章为实验设计与结果分析，第六章为总结与展望。第五章基于仿真实验和对比算法验证所提方法在交期、拖期、维护成本和系统稳定性等指标上的表现；第六章总结本文工作，分析模型和实验中仍存在的限制，并提出未来可进一步研究的方向。",
]

new_refs = [
    'H. Lasi, P. Fettke, H.-G. Kemper, et al., "Industry 4.0," Business & Information Systems Engineering, vol. 6, no. 4, pp. 239-242, Aug. 2014.',
    'J. Lee, B. Bagheri, and H.-A. Kao, "A Cyber-Physical Systems architecture for Industry 4.0-based manufacturing systems," Manufacturing Letters, vol. 3, pp. 18-23, Jan. 2015.',
    'J. Lee, J. Lapira, B. Bagheri, et al., "Recent advances and trends in predictive manufacturing systems in big data environment," Manufacturing Letters, vol. 1, no. 1, pp. 38-41, Oct. 2013.',
    'J. Lee, F. Wu, W. Zhao, et al., "Prognostics and health management design for rotary machinery systems - Reviews, methodology and applications," Mechanical Systems and Signal Processing, vol. 42, no. 1-2, pp. 314-334, Jan. 2014.',
    'A. K. S. Jardine, D. Lin, and D. Banjevic, "A review on machinery diagnostics and prognostics implementing condition-based maintenance," Mechanical Systems and Signal Processing, vol. 20, no. 7, pp. 1483-1510, Oct. 2006.',
    'T. Zonta, C. A. da Costa, R. da Rosa Righi, et al., "Predictive maintenance in the Industry 4.0: A systematic literature review," Computers & Industrial Engineering, vol. 150, p. 106889, Dec. 2020.',
    'T. P. Carvalho, F. A. A. M. N. Soares, R. Vita, et al., "A systematic literature review of machine learning methods applied to predictive maintenance," Computers & Industrial Engineering, vol. 137, p. 106024, Nov. 2019.',
    'J. Z. Sikorska, M. Hodkiewicz, and L. Ma, "Prognostic modelling options for remaining useful life estimation by industry," Mechanical Systems and Signal Processing, vol. 25, no. 5, pp. 1803-1836, Jul. 2011.',
    'X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, "Remaining useful life estimation - A review on the statistical data driven approaches," European Journal of Operational Research, vol. 213, no. 1, pp. 1-14, Aug. 2011.',
    'Y. Lei, N. Li, L. Guo, et al., "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," Mechanical Systems and Signal Processing, vol. 104, pp. 799-834, May 2018.',
    'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, Nov. 1997.',
    'K. Cho, B. van Merriënboer, C. Gulcehre, et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 2014, pp. 1724-1734.',
    'J. Chung, C. Gulcehre, K. Cho, et al., "Empirical evaluation of gated recurrent neural networks on sequence modeling," arXiv preprint arXiv:1412.3555, 2014.',
    'F. O. Heimes, "Recurrent neural networks for remaining useful life estimation," in Proceedings of the 2008 International Conference on Prognostics and Health Management, Denver, CO, USA, 2008, pp. 1-6.',
    'S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, "Long short-term memory network for remaining useful life estimation," in Proceedings of the 2017 IEEE International Conference on Prognostics and Health Management, Dallas, TX, USA, 2017, pp. 88-95.',
    'X. Li, Q. Ding, and J.-Q. Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks," Reliability Engineering & System Safety, vol. 172, pp. 1-11, Apr. 2018.',
    'J. Zhang, P. Wang, R. Yan, et al., "Long short-term memory for machine remaining life prediction," Journal of Manufacturing Systems, vol. 48, pp. 78-86, Jul. 2018.',
    'M. L. Pinedo, Scheduling: Theory, Algorithms, and Systems, 5th ed. Cham, Switzerland: Springer, 2016.',
    'S. S. Panwalkar and W. Iskander, "A survey of scheduling rules," Operations Research, vol. 25, no. 1, pp. 45-61, Feb. 1977.',
    'A. P. J. Vepsalainen and T. E. Morton, "Priority rules for job shops with weighted tardiness costs," Management Science, vol. 33, no. 8, pp. 1035-1047, Aug. 1987.',
    'P. Brucker and R. Schlie, "Job-shop scheduling with multi-purpose machines," Computing, vol. 45, no. 4, pp. 369-375, Dec. 1990.',
    'P. Brandimarte, "Routing and scheduling in a flexible job shop by tabu search," Annals of Operations Research, vol. 41, no. 3, pp. 157-183, Sep. 1993.',
    'M. Mastrolilli and L. M. Gambardella, "Effective neighbourhood functions for the flexible job shop problem," Journal of Scheduling, vol. 3, no. 1, pp. 3-20, Jan. 2000.',
    'I. Kacem, S. Hammadi, and P. Borne, "Approach by localization and multiobjective evolutionary optimization for flexible job-shop scheduling problems," IEEE Transactions on Systems, Man, and Cybernetics, Part C, vol. 32, no. 1, pp. 1-13, Feb. 2002.',
    'J. Gao, L. Sun, and M. Gen, "A hybrid genetic and variable neighborhood descent algorithm for flexible job shop scheduling problems," Computers & Operations Research, vol. 35, no. 9, pp. 2892-2907, Sep. 2008.',
    'I. A. Chaudhry and A. A. Khan, "A research survey: Review of flexible job shop scheduling techniques," International Transactions in Operational Research, vol. 23, no. 3, pp. 551-591, May 2016.',
    'S. Dauzère-Pérès, W. Roux, and J.-B. Lasserre, "Multi-resource shop scheduling with resource flexibility," European Journal of Operational Research, vol. 107, no. 2, pp. 289-305, Jun. 1998.',
    'D. Ouelhadj and S. Petrovic, "A survey of dynamic scheduling in manufacturing systems," Journal of Scheduling, vol. 12, no. 4, pp. 417-431, Aug. 2009.',
    'G. E. Vieira, J. W. Herrmann, and E. Lin, "Rescheduling manufacturing systems: A framework of strategies, policies, and methods," Journal of Scheduling, vol. 6, no. 1, pp. 39-62, Jan. 2003.',
    'C. R. Cassady and E. Kutanoglu, "Integrating preventive maintenance planning and production scheduling for a single machine," IEEE Transactions on Reliability, vol. 54, no. 2, pp. 304-309, Jun. 2005.',
    'R. Ruiz, C. García-Díaz, and J. C. Maroto, "Considering scheduling and preventive maintenance in the flowshop sequencing problem," Computers & Operations Research, vol. 34, no. 11, pp. 3314-3330, Nov. 2007.',
    'M.-C. Fitouhi and M. Nourelfath, "Integrating noncyclical preventive maintenance scheduling and production planning for multi-state systems," Reliability Engineering & System Safety, vol. 121, pp. 175-186, Jan. 2014.',
    'L. Xiao, S. Song, X. Chen, et al., "Joint optimization of production scheduling and machine group preventive maintenance," Reliability Engineering & System Safety, vol. 146, pp. 68-78, Feb. 2016.',
    'R. S. Sutton, D. Precup, and S. Singh, "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning," Artificial Intelligence, vol. 112, no. 1-2, pp. 181-211, Aug. 1999.',
    'C. J. C. H. Watkins and P. Dayan, "Q-learning," Machine Learning, vol. 8, no. 3-4, pp. 279-292, May 1992.',
    'H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 30, no. 1, 2016, pp. 2094-2100.',
]


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="docx_full_refs_") as td:
        tdir = Path(td)
        with ZipFile(SRC, "r") as zin:
            zin.extractall(tdir)

        doc_xml = tdir / "word/document.xml"
        tree = ET.parse(doc_xml)
        root = tree.getroot()
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("w:body not found")

        body_template = direct_paragraph_by_contains(body, "传统的工业生产为了降低成本通常采用预先大量制造再大量贩售")
        ref_template = direct_paragraph_by_contains(body, "Kagermann")

        replace_between(body, "引言", "研究背景", [make_paragraph(p, body_template) for p in intro_paras])
        replace_between(body, "研究背景", "研究现状", [make_paragraph(p, body_template) for p in background_paras])
        replace_between(body, "研究现状", "论文结构", [make_paragraph(p, body_template) for p in status_paras])
        replace_between(body, "论文结构", "基础理论与背景知识", [make_paragraph(p, body_template) for p in structure_paras])

        ack_idx = direct_child_index_by_text(body, "致谢")
        children = list(body)
        ref_paras = [make_paragraph(ref, ref_template, superscript_cites=False) for ref in new_refs]
        body[:] = children[:ack_idx] + ref_paras + children[ack_idx:]

        tree.write(doc_xml, encoding="UTF-8", xml_declaration=True)

        with ZipFile(OUT, "w", ZIP_DEFLATED) as zout:
            for path in tdir.rglob("*"):
                if path.is_file():
                    zout.write(path, path.relative_to(tdir).as_posix())

    print(OUT)


if __name__ == "__main__":
    main()
