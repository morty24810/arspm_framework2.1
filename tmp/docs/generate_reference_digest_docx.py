# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path("/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1")
OUT = ROOT / "output/doc/本科毕业设计_第一章参考文献整理.docx"


ENTRIES = [
    (1, "Kagermann等关于工业4.0战略实施的报告，主要说明工业4.0通过信息物理系统、工业互联网、数据互联和智能决策推动制造系统转型。", "该报告不是提出具体算法，而是提出工业4.0的总体技术路线和产业发展观点，强调制造资源互联、生产过程透明化和决策智能化。", "第一章引言中已经用该文献说明本文研究的宏观背景，即机器状态和生产调度需要在智能制造框架下被统一感知和决策。"),
    (2, "Luo研究动态柔性作业车间中插入新作业后的调度问题，关注生产系统在动态扰动下如何重新安排工序。", "文献将动态柔性作业车间调度建模为强化学习问题，并采用深度强化学习学习调度规则选择或动作价值，从而应对新作业插入带来的状态变化。", "第一章用该文献说明动态柔性作业车间调度已经可以通过深度强化学习处理，为本文把调度问题建模为学习型决策问题提供直接依据。"),
    (3, "Luo、Zhang和Fan进一步研究动态柔性作业车间的多目标调度问题，强调调度目标并不只有单一完工时间或拖期。", "该文献提出双层级深度Q网络思想，通过高层目标选择和低层调度规则选择处理多目标之间的权衡。", "第一章提到本文采用Two-Hierarchy Deep Q-Network时，实际上延续了该文献中高层目标和低层规则分离的思路。"),
    (4, "Ren和Liu研究基于MachineRank和强化学习的动态柔性作业车间调度，重点在于动态环境下机器重要性和调度动作选择。", "文献结合机器排序评价和强化学习方法，利用机器状态与作业信息辅助调度决策。", "第一章引用该文献，是为了说明近年的动态调度研究已经开始把机器状态特征纳入调度策略，而本文进一步加入健康状态和RUL信息。"),
    (5, "Zhang等研究多智能体制造系统中的动态作业车间调度，关注多主体协同决策问题。", "该文献将作业、机器或调度单元建模为多智能体系统，通过深度强化学习实现动态调度协同。", "第一章用该文献说明复杂制造系统中的调度问题可以采用多智能体或分布式智能决策思想，支撑本文将维护器和调度器拆分为不同决策模块。"),
    (6, "Ding等研究生产调度与维护的自适应实时协同问题，重点是将RUL预测与多智能体深度强化学习结合。", "文献通过RUL预测提供设备健康信息，再由强化学习智能体进行调度和维护决策，以减少故障和生产损失。", "第一章中该文献是最贴近本文主题的依据，用于说明RUL预测、维护决策和实时调度可以放入统一框架。"),
    (7, "Mallioris等对工业4.0背景下的预测性维护进行跨行业系统映射，说明预测性维护在不同产业中的应用范围。", "该文献采用系统综述和映射方法，总结预测性维护的技术组成、行业应用和研究趋势。", "第一章用该文献说明维护方式正从固定周期或故障后维修转向基于状态和预测的主动维护。"),
    (8, "Sutton和Barto的著作系统介绍强化学习的基本理论，包括马尔可夫决策过程、价值函数、策略、奖励和探索利用。", "该书提出的是强化学习理论框架，不是针对某个制造问题的具体解法。", "第一章以该文献作为本文强化学习建模的理论来源，后续维护智能体和调度智能体都建立在状态-动作-奖励框架上。"),
    (9, "Mnih等提出深度Q网络，使强化学习能够利用深度神经网络处理高维状态输入。", "文献将Q-learning与深度神经网络、经验回放和目标网络结合，用于学习动作价值函数。", "第一章用该文献说明DQN类方法的来源，本文维护端DDQN和调度端THDQN中的价值函数学习都继承了这一思想。"),
    (10, "Schulman等提出近端策略优化算法，关注策略梯度方法训练不稳定的问题。", "PPO通过限制新旧策略之间的更新幅度，避免策略更新过大导致性能崩塌。", "第一章引用该文献，是为了说明本文PPO调度器属于直接学习状态到调度规则概率分布的策略优化方法。"),
    (11, "Silver和Veness提出POMCP，用于大规模部分可观测马尔可夫决策过程中的在线规划。", "文献将蒙特卡洛树搜索与粒子信念表示结合，在当前信念状态下通过模拟搜索选择动作。", "第一章中该文献用于说明本文POMCP维护器的理论来源，即在健康状态存在不确定性时采用在线规划作为对比路径。"),
    (12, "Hagmeyer等讨论公开预测与诊断数据集的构建，强调数据场景对工业PHM研究的重要性。", "文献从数据集公开性、退化过程、故障模式和工业相关性等角度分析数据集设计。", "第一章用该文献说明本文RUL预测任务依赖公开工业退化数据，数据来源和数据场景会影响模型验证。"),
    (13, "PrognosticsHSE提供从预防性维护到预测性维护的数据集，包含与设备状态和维护相关的观测数据。", "该数据集不是方法论文，而是为预测性维护和RUL建模提供实验数据基础。", "第一章引用该数据集，是为了说明本文RUL预测模块的实验数据来源。"),
    (14, "Lasi等概述Industry 4.0概念，强调制造系统数字化、网络化和智能化的发展方向。", "文献提出工业4.0的概念框架和关键组成，包括信息系统集成、自动化与智能制造。", "第一章研究背景部分使用该文献补充工业4.0技术发展史，说明本文生产调度研究处于智能制造语境下。"),
    (15, "Oks等对工业4.0背景下的信息物理系统进行综述、分类和展望。", "文献通过综述方法归纳CPS的组成、应用场景和研究方向，强调物理设备与数字系统的耦合。", "第一章借该文献说明设备状态、传感数据和决策模型之间需要建立连接，这与本文把健康状态转化为调度状态变量一致。"),
    (16, "Kusiak讨论数字制造中的预测模型，关注预测模型在制造研究和应用中的作用。", "文献从研究、应用和未来趋势角度说明预测模型如何支持数字制造中的生产计划、质量控制和维护决策。", "第一章用该文献说明预测性制造的核心观点：在故障或性能下降发生前利用预测结果调整生产和维护策略。"),
    (17, "Huang等综述预测性维护中的PHM研究，覆盖故障诊断、健康评估、寿命预测和维护决策。", "文献通过综述方式总结PHM面向预测性维护的流程和关键技术。", "第一章中该文献支撑了从传统维护转向基于状态和预测的维护方式这一论述。"),
    (18, "Benhanifia等系统综述制造业中的预测性维护实践，强调预测性维护在真实制造场景中的部署问题。", "文献总结制造业预测性维护的技术路线、应用限制和实践挑战。", "第一章引用该文献，是为了说明RUL预测应服务于维护窗口、备件准备和生产排程，而不是停留在预测误差评价。"),
    (19, "Zonta等系统综述工业4.0中的预测性维护，归纳数据采集、数据分析、预测模型和维护策略。", "文献采用系统文献综述方法，梳理预测性维护的组成环节与工业4.0技术之间的联系。", "第一章用该文献说明预测性维护一般包括数据采集、特征构造、健康识别、RUL预测和维护优化等环节。"),
    (20, "Carvalho等综述机器学习方法在预测性维护中的应用。", "文献比较监督学习、无监督学习和深度学习等方法在故障诊断与维护预测中的使用方式。", "第一章借该文献说明机器学习已经成为预测性维护的重要方法基础，并引出本文使用GRU进行RUL预测。"),
    (21, "Ferreira和Goncalves综述基于机器学习的RUL预测及其挑战。", "文献从数据质量、特征选择、模型泛化和工业部署等角度总结RUL预测难点。", "第一章用该文献说明数据驱动RUL方法适合传感器数据丰富但物理机理难以准确建模的场景。"),
    (22, "Li等综述物理信息融合的数据驱动RUL预测，关注机理知识与数据模型结合。", "文献提出物理信息数据驱动方法能够缓解纯数据模型可解释性不足和跨工况泛化困难的问题。", "第一章引用该文献，是为了说明RUL建模不只有纯数据驱动，也可以结合统计、物理和机理信息；本文则选择数据驱动GRU作为实现路径。"),
    (23, "Lei等系统综述机械健康预测从数据采集到RUL预测的完整流程。", "文献归纳信号采集、特征提取、健康指标构造、预测模型和评估方法。", "第一章用该文献说明RUL预测效果依赖信息整合、时序特征提取和复杂工况适应能力。"),
    (24, "Hochreiter和Schmidhuber提出LSTM，用于解决普通循环神经网络难以学习长时依赖的问题。", "LSTM通过输入门、遗忘门和输出门控制信息保留与更新，缓解梯度消失。", "第一章引用该文献，是为了说明深度时序模型的发展基础，并与本文使用的GRU形成技术背景。"),
    (25, "Cho等提出GRU相关的编码器-解码器结构，用于序列建模任务。", "GRU通过更新门和重置门简化循环神经网络结构，在保留记忆能力的同时减少参数量。", "第一章中该文献是本文选择GRU作为RUL预测器的基础来源。"),
    (26, "Chung等对GRU、LSTM等门控循环网络进行经验比较。", "文献通过实验比较不同门控循环结构在序列建模任务中的表现。", "第一章引用该文献，是为了说明GRU在若干序列预测任务中具有较好效果和较低复杂度。"),
    (27, "Lin等研究基于注意力GRU的RUL预测方法。", "文献在GRU结构中引入注意力机制，以增强模型对关键退化时间片段或特征的关注。", "第一章用该文献说明RUL领域已有GRU及其改进模型，本文采用GRU提供健康状态估计具有合理依据。"),
    (28, "Sahu和Rai研究基于LSTM和C-MMPE特征的滚动轴承RUL预测。", "文献结合特征构造和LSTM深度序列模型解决轴承退化寿命预测问题。", "第一章引用该文献，是为了说明LSTM类深度时序模型已经被用于具体设备的RUL预测。"),
    (29, "Peng等研究涡扇发动机RUL预测中的深度特征提取与融合。", "文献通过深度学习提取多源退化特征，并进行特征融合以提升RUL预测性能。", "第一章中该文献用于说明CNN或深度特征融合方法也可用于从传感序列中学习退化模式。"),
    (30, "Zhang等提出嵌入式注意力并行网络预测机器剩余寿命。", "文献通过并行网络结构和注意力机制增强对不同退化特征的表达能力。", "第一章引用该文献，是为了说明近年来RUL预测已从单一循环网络发展到注意力和并行结构等深度序列模型。"),
    (31, "Pinedo的调度著作系统介绍调度理论、算法和系统，是生产调度领域的基础文献。", "该书建立机器调度、作业车间调度、目标函数和约束表达等经典理论框架。", "第一章用该文献说明传统调度研究主要关注有限资源下的工序排序、机器分配和目标优化。"),
    (32, "Panwalkar和Iskander综述调度规则，是派工规则研究的经典文献。", "文献归纳最短加工时间、最早交期、关键比率等规则，并讨论它们在不同调度目标下的适用性。", "第一章引用该文献，是为了说明本文调度动作空间中的规则选择来自传统调度规则体系。"),
    (33, "Vepsalainen和Morton研究带加权拖期成本的作业车间优先规则。", "文献围绕拖期成本设计和分析优先级规则，强调交期压力和权重对调度选择的影响。", "第一章用该文献说明传统规则可以反映交期和拖期压力，但单独使用时难以处理健康和维护耦合。"),
    (34, "Brucker和Schlie研究多用途机器条件下的作业车间调度。", "文献讨论一道工序可由多台机器加工时的调度建模问题，是柔性作业车间思想的重要早期基础。", "第一章引用该文献，是为了说明FJSP相较传统作业车间的关键变化在于机器选择自由度。"),
    (35, "Brandimarte研究利用禁忌搜索处理柔性作业车间中的路径选择和调度。", "文献采用禁忌搜索在工序排序和机器选择组合空间中寻找较优解。", "第一章用该文献说明早期FJSP常依赖启发式或元启发式算法求解。"),
    (36, "Mastrolilli和Gambardella研究FJSP中的有效邻域结构。", "文献通过设计邻域搜索函数改进局部搜索效率，从而提升FJSP求解质量。", "第一章引用该文献，是为了说明元启发式方法在静态或准静态FJSP中具有重要作用。"),
    (37, "Kacem等研究FJSP的多目标进化优化。", "文献通过多目标进化方法同时考虑多个调度目标，例如完工时间、负载平衡等。", "第一章用该文献说明FJSP天然具有多目标特性，这与本文后续高层目标选择和成本权衡相关。"),
    (38, "Xue等研究带并行批处理机器的柔性作业车间调度问题。", "文献提出增强型多种群遗传算法，用于处理更复杂机器环境下的FJSP优化。", "第一章引用该文献，是为了说明近年FJSP研究仍在通过改进元启发式算法扩展问题场景。"),
    (39, "Dauzère-Pérès等对柔性作业车间调度问题进行综述。", "文献系统总结FJSP的模型、约束、求解方法和研究趋势。", "第一章用该文献说明FJSP是典型NP-hard问题，难点在于机器选择、工序排序和多约束耦合。"),
    (40, "Li等综述集成柔性作业车间调度问题。", "文献关注FJSP与运输、维护、能耗或其他生产环节集成后的复杂调度模型。", "第一章引用该文献，是为了说明本文并非标准FJSP，而是考虑维护和健康状态的集成调度问题。"),
    (41, "Ngwu等综述强化学习在动态作业车间调度中的应用。", "文献总结AI驱动调度方法在动态扰动、实时决策和现代制造中的研究进展。", "第一章用该文献说明动态调度需要具备持续重调度能力，而强化学习是近年来的重要研究路径。"),
    (42, "An等研究新作业插入和机器预防性维护条件下的多目标柔性作业车间重调度。", "文献将新作业插入和机器维护共同纳入重调度模型，并处理多目标优化问题。", "第一章引用该文献，是为了说明动态作业到达和维护活动会共同改变调度可行性，静态排程容易失效。"),
    (43, "Ghaleb等研究带机器退化和状态维护的实时生产调度与维护计划集成。", "文献在柔性作业车间中同时考虑生产排程、机器退化和基于状态的维护。", "第一章用该文献支撑本文核心观点：维护不应作为外部补丁，而应与生产调度联合建模。"),
    (44, "Pal等研究柔性作业车间中的多智能体集成调度与维护计划。", "文献采用多智能体系统协调生产调度和维护规划，处理不同决策主体之间的交互。", "第一章引用该文献，是为了说明多智能体思想可用于生产和维护协同，与本文维护器和调度器分离设计相呼应。"),
    (45, "Zhang等提出可修复多单元系统的预防性维护和任务调度集成框架。", "文献将维护任务与生产任务或系统任务统一安排，解决维修资源和任务时序协调问题。", "第一章用该文献说明维护活动会占用资源并影响后续任务执行，需要进入统一调度框架。"),
    (46, "Wang等研究柔性制造环境中生产调度、预防性维护和节能的集成协调。", "文献同时考虑生产、维护和能源目标，构建多目标协同优化框架。", "第一章引用该文献，是为了说明生产维护协同还可能扩展到能耗等系统目标，本文则聚焦成本、拖期和健康状态。"),
    (47, "Sutton、Precup和Singh提出MDP与半MDP之间的时间抽象框架。", "文献提出选项框架，将复杂序贯决策拆分为不同层级的子策略或宏动作。", "第一章用该文献说明本文分层维护和THDQN调度的理论动机，即通过层级分解决策降低单一策略负担。"),
    (48, "Watkins和Dayan提出Q-learning，是基于价值函数的强化学习基础算法。", "Q-learning通过贝尔曼最优方程迭代学习状态-动作价值，在不完全知道环境模型时寻找最优策略。", "第一章引用该文献，是为了说明DQN、DDQN和THDQN的价值学习根源。"),
    (49, "van Hasselt、Guez和Silver提出Double DQN。", "文献将动作选择和动作价值评估解耦，降低传统DQN最大化操作带来的价值过估计。", "第一章用该文献说明本文DDQN维护器和THDQN调度器采用Double DQN更新思想，以提升训练稳定性。"),
]


def para(text: str, style: str = "Normal", bold: bool = False, center: bool = False) -> str:
    jc = '<w:jc w:val="center"/>' if center else ""
    pstyle = f'<w:pStyle w:val="{style}"/>' if style else ""
    rpr = "<w:b/>" if bold else ""
    return (
        "<w:p><w:pPr>"
        f"{pstyle}{jc}"
        "</w:pPr><w:r><w:rPr>"
        f"{rpr}<w:rFonts w:ascii=\"SimSun\" w:eastAsia=\"宋体\" w:hAnsi=\"SimSun\"/>"
        "</w:rPr><w:t xml:space=\"preserve\">"
        f"{escape(text)}"
        "</w:t></w:r></w:p>"
    )


def build_document_xml() -> str:
    body = []
    body.append(para("第一章参考文献整理", style="Title", bold=True, center=True))
    body.append(para("整理依据：本文《本科毕业设计 傅懋杰 2.0.docx》第一章及其参考文献表。整理顺序完全按照论文参考文献编号展开，重点说明每篇文献的主要内容、解决问题的方法或观点，以及它在第一章论述中的作用。"))
    body.append(para("一、按论文引用顺序整理", style="Heading1", bold=True))
    for no, content, method, relation in ENTRIES:
        body.append(para(f"文献[{no}]", style="Heading2", bold=True))
        body.append(para(f"主要内容：{content}"))
        body.append(para(f"方法或观点：{method}"))
        body.append(para(f"与第一章的关系：{relation}"))
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:body>"
        + "".join(body)
        + sect
        + "</w:body></w:document>"
    )


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

WORD_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="SimSun" w:eastAsia="宋体" w:hAnsi="SimSun"/><w:sz w:val="24"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto" w:after="120"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:line="360" w:lineRule="auto" w:after="120"/></w:pPr><w:rPr><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:spacing w:before="180" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
</w:styles>
"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>第一章参考文献整理</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", WORD_RELS)
        z.writestr("word/document.xml", build_document_xml())
        z.writestr("word/styles.xml", STYLES)
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", APP_XML)
    print(OUT)


if __name__ == "__main__":
    main()
