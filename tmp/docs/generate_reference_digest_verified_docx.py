# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path("/Users/Morty/Desktop/毕业设计/Code/RUL_GRU/arspm_framework2.1")
OUT = ROOT / "output/doc/本科毕业设计_第一章参考文献逐篇核查整理.docx"


ENTRIES = [
    {
        "n": 1,
        "source": "官方报告/出版信息核查；原文为德国工业4.0工作组最终报告。",
        "keywords": "Industry 4.0；智能制造；信息物理系统；制造业数字化。",
        "actual": "该报告围绕德国工业4.0战略展开，说明制造业应通过信息物理系统、网络化生产和数据集成实现产业升级。它讨论的不是单一算法，而是智能制造体系的战略框架和实施建议。",
        "method": "以政策报告和产业路线图形式提出工业4.0架构、标准化、组织和实施路径。",
        "conclusion": "制造系统未来需要实现设备、生产过程和信息系统的深度互联，并通过实时数据支持柔性生产和智能决策。",
        "use": "第一章用该文献引出智能制造背景，说明机器状态和调度决策需要放入工业4.0的数据互联框架中讨论。",
    },
    {
        "n": 2,
        "source": "Crossref/DOI核查：10.1016/j.asoc.2020.106208。",
        "keywords": "动态柔性作业车间；新作业插入；深度强化学习；调度规则。",
        "actual": "该文献研究动态柔性作业车间中不断插入新作业时的调度问题，目标是在动态环境下快速选择合适的派工规则。",
        "method": "将动态调度过程构造为强化学习问题，使用深度强化学习学习状态到调度动作的映射，以适应新作业到达导致的状态变化。",
        "conclusion": "深度强化学习能够在动态作业插入环境中形成比固定规则更具自适应性的调度策略。",
        "use": "第一章引用该文献说明动态FJSP已有DRL求解路径，本文进一步在动态调度中加入RUL和维护耦合。",
    },
    {
        "n": 3,
        "source": "Crossref/DOI核查：10.1016/j.cie.2021.107489。",
        "keywords": "动态多目标调度；柔性作业车间；THDQN；深度强化学习。",
        "actual": "该文献研究动态柔性作业车间的多目标调度问题，关注不同生产目标之间的权衡。",
        "method": "提出双层级深度Q网络，由高层选择当前优化目标，低层根据目标选择具体调度规则。",
        "conclusion": "将目标选择和规则选择分层处理，有助于动态环境下实现多目标调度权衡。",
        "use": "第一章和后文THDQN调度器均沿用该文献的高层目标-低层规则结构，作为本文调度端分层设计依据。",
    },
    {
        "n": 4,
        "source": "Crossref/DOI核查：10.1038/s41598-024-79593-8。",
        "keywords": "MachineRank；动态调度；强化学习；机器优先级。",
        "actual": "该文献研究动态柔性作业车间调度，强调机器在生产网络中的重要性评价对调度决策的作用。",
        "method": "结合MachineRank算法和强化学习，利用机器相关信息辅助动态调度动作选择。",
        "conclusion": "把机器状态或机器重要性纳入调度状态，可以提升动态调度策略对复杂车间环境的适应能力。",
        "use": "第一章用该文献说明近年调度研究已开始关注机器侧信息，本文进一步把机器健康和RUL纳入状态。",
    },
    {
        "n": 5,
        "source": "Crossref/DOI核查：10.1016/j.rcim.2022.102412。",
        "keywords": "多智能体制造系统；动态作业车间；深度强化学习；协同调度。",
        "actual": "该文献研究多智能体制造系统中的动态作业车间调度问题，强调多个生产主体之间的交互。",
        "method": "使用深度强化学习对多智能体制造系统进行建模，使不同主体在动态环境中协同形成调度决策。",
        "conclusion": "多智能体强化学习适合处理制造系统中多个决策单元之间的动态协同。",
        "use": "第一章引用该文献支撑本文维护智能体和调度智能体分离建模的合理性。",
    },
    {
        "n": 6,
        "source": "Crossref/DOI核查：10.1016/j.ress.2025.111394。",
        "keywords": "RUL预测；生产维护协同；多智能体深度强化学习；实时调度。",
        "actual": "该文献研究将RUL预测、实时生产调度和维护决策结合的自适应调度维护问题。",
        "method": "利用RUL预测表示设备退化风险，并通过多智能体深度强化学习协调生产和维护动作。",
        "conclusion": "RUL信息进入决策层后，可以支持更主动的维护安排并降低故障对生产调度的影响。",
        "use": "第一章用该文献直接说明本文主题的可行性，即RUL预测、维护和调度可以统一建模。",
    },
    {
        "n": 7,
        "source": "Crossref/DOI核查：10.1016/j.cirpj.2024.02.003。",
        "keywords": "预测性维护；工业4.0；跨行业映射；系统综述。",
        "actual": "该文献系统映射预测性维护在工业4.0中的多行业应用，梳理技术、场景和应用趋势。",
        "method": "采用系统文献映射方法，对预测性维护研究按行业、方法和应用环节进行归纳。",
        "conclusion": "预测性维护已经成为工业4.0中的关键应用方向，并逐步从设备层预测扩展到系统层决策支持。",
        "use": "第一章用该文献说明维护方式正在由被动维护转向基于状态和寿命预测的主动维护。",
    },
    {
        "n": 8,
        "source": "教材原文/出版信息核查：Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed., MIT Press, 2018。",
        "keywords": "强化学习；MDP；价值函数；策略；奖励。",
        "actual": "该书系统阐述强化学习基本理论，包括智能体、环境、状态、动作、奖励、价值函数和策略学习。",
        "method": "以马尔可夫决策过程为基础，建立动态规划、蒙特卡洛、时序差分和函数逼近等方法体系。",
        "conclusion": "强化学习适合描述智能体通过交互学习长期回报最大化策略的问题。",
        "use": "第一章引用该书作为本文维护和调度智能体建模的理论基础。",
    },
    {
        "n": 9,
        "source": "Crossref/DOI核查：10.1038/nature14236。",
        "keywords": "DQN；深度强化学习；经验回放；目标网络。",
        "actual": "该文献提出深度Q网络，使智能体能够从高维输入中学习动作价值并达到人类水平控制表现。",
        "method": "将Q-learning与卷积神经网络、经验回放和目标网络结合，稳定训练深度动作价值函数。",
        "conclusion": "深度神经网络可以作为Q函数近似器，扩展强化学习处理高维复杂状态的能力。",
        "use": "第一章用该文献说明本文DQN、DDQN和THDQN价值函数学习的深度强化学习来源。",
    },
    {
        "n": 10,
        "source": "arXiv原始页面核查：arXiv:1707.06347。",
        "keywords": "PPO；策略梯度；裁剪目标；稳定训练。",
        "actual": "该文献提出近端策略优化算法，目标是改进策略梯度方法训练不稳定、更新幅度过大的问题。",
        "method": "通过裁剪概率比或近端约束限制新旧策略差异，在多轮小批量更新中优化策略。",
        "conclusion": "PPO在实现简单的同时具有较好的训练稳定性和样本效率，因此广泛用于策略优化任务。",
        "use": "第一章用该文献说明本文PPO调度器的算法来源，即直接学习状态到调度规则分布。",
    },
    {
        "n": 11,
        "source": "NeurIPS论文页面/原始会议信息核查：Monte-Carlo Planning in Large POMDPs, NeurIPS 2010。",
        "keywords": "POMCP；POMDP；蒙特卡洛树搜索；粒子信念。",
        "actual": "该文献研究大规模部分可观测决策问题中的在线规划，解决显式维护完整信念状态成本过高的问题。",
        "method": "结合蒙特卡洛树搜索和粒子滤波信念表示，在仿真过程中估计动作价值。",
        "conclusion": "POMCP能够在较大POMDP中进行有效在线规划，而不需要完整枚举信念空间。",
        "use": "第一章用该文献说明本文POMCP维护器作为在线规划型对比方法的理论来源。",
    },
    {
        "n": 12,
        "source": "Crossref/DOI核查：10.36001/ijphm.2021.v12i2.3087。",
        "keywords": "PHM数据集；公开数据；故障预测；诊断。",
        "actual": "该文献讨论面向PHM研究的公开数据集构建问题，强调数据场景覆盖对预测和诊断研究的重要性。",
        "method": "从数据集可用性、故障覆盖、运行到失效过程和工业相关性角度评价公开数据资源。",
        "conclusion": "高质量公开数据集是PHM模型开发和验证的重要前提，数据场景比单纯样本量更关键。",
        "use": "第一章引用该文献说明本文RUL预测任务依赖公开工业退化数据，数据来源需要与工业场景相关。",
    },
    {
        "n": 13,
        "source": "Kaggle数据集页面核查：Preventive to Predictive Maintenance。",
        "keywords": "预测性维护数据集；设备状态；传感数据；维护预测。",
        "actual": "该条是数据集来源，提供用于从预防性维护转向预测性维护的数据记录。",
        "method": "数据集本身不提出算法，而是提供设备运行或维护相关样本，供建模和验证使用。",
        "conclusion": "该数据集可作为预测性维护/RUL相关实验的数据基础。",
        "use": "第一章引用该数据集说明本文RUL预测和实验数据构造的来源。",
    },
    {
        "n": 14,
        "source": "Crossref/DOI核查：10.1007/s12599-014-0334-4。",
        "keywords": "Industry 4.0；智能工厂；数字化；生产系统。",
        "actual": "该文献概述工业4.0概念、背景和关键特征，是工业4.0研究中的基础概念文献。",
        "method": "以概念综述方式说明工业4.0如何通过数字化、网络化和自动化改变制造系统。",
        "conclusion": "工业4.0将推动生产系统向高度集成、柔性和智能方向发展。",
        "use": "第一章用它补充工业4.0技术发展史，支撑本文智能制造背景描述。",
    },
    {
        "n": 15,
        "source": "OpenAlex摘要/Crossref DOI核查：10.1007/s10796-022-10252-x。",
        "keywords": "CPS；工业4.0；系统分类；数字化转型。",
        "actual": "该文献对工业4.0语境下的信息物理系统进行综述、分类和展望。",
        "method": "通过文献综述和分类框架归纳CPS的技术组成、应用类别和未来方向。",
        "conclusion": "CPS能够统一物理设备、数据通信和信息系统，是制造业数字化转型的重要基础。",
        "use": "第一章用该文献说明设备状态、传感数据和决策模型连接起来后，健康数据才能成为调度状态变量。",
    },
    {
        "n": 16,
        "source": "Crossref/DOI核查：10.1080/00207543.2022.2122620。",
        "keywords": "预测模型；数字制造；数据驱动；未来展望。",
        "actual": "该文献讨论预测模型在数字制造中的研究、应用和未来发展。",
        "method": "从制造过程预测、质量控制、维护和生产决策等角度总结预测模型的作用。",
        "conclusion": "预测模型能够将制造数据转化为提前干预和优化决策的依据。",
        "use": "第一章用它说明预测性制造的观点，即在故障前利用预测结果调整生产和维护策略。",
    },
    {
        "n": 17,
        "source": "Crossref/DOI核查：10.1016/j.jmsy.2024.05.021。",
        "keywords": "PHM；预测性维护；健康管理；综述。",
        "actual": "该文献综述面向预测性维护的PHM研究，涵盖健康监测、诊断、预测和维护决策。",
        "method": "以综述方式整理PHM技术链条及其在预测性维护中的应用。",
        "conclusion": "PHM为预测性维护提供从状态感知到维护决策的完整技术支撑。",
        "use": "第一章用它说明维护方式从事后维修、定期维护转向基于状态和预测的维护。",
    },
    {
        "n": 18,
        "source": "Crossref/DOI核查：10.1016/j.iswa.2025.200501。",
        "keywords": "预测性维护实践；制造业；系统综述；部署挑战。",
        "actual": "该文献系统综述制造业预测性维护实践，关注方法落地和应用约束。",
        "method": "通过系统综述归纳制造业预测性维护的技术流程、场景和限制。",
        "conclusion": "预测性维护需要与生产管理、维护资源和工业部署条件结合，不能只停留在模型预测。",
        "use": "第一章用它说明RUL预测应服务于维护窗口、备件准备和生产排程。",
    },
    {
        "n": 19,
        "source": "Crossref/DOI核查：10.1016/j.cie.2020.106889。",
        "keywords": "预测性维护；工业4.0；系统文献综述；机器学习。",
        "actual": "该文献系统综述工业4.0中的预测性维护研究，梳理数据、模型和应用环节。",
        "method": "采用系统文献综述方法，对预测性维护研究按技术、行业和流程进行分类。",
        "conclusion": "预测性维护通常包含数据采集、预处理、诊断预测和维护优化等环节。",
        "use": "第一章用它说明预测性维护流程结构，并引出RUL预测在其中的位置。",
    },
    {
        "n": 20,
        "source": "Crossref/DOI核查：10.1016/j.cie.2019.106024。",
        "keywords": "机器学习；预测性维护；故障诊断；系统综述。",
        "actual": "该文献综述机器学习方法在预测性维护中的应用。",
        "method": "比较不同机器学习方法在故障检测、诊断和预测任务中的适用性。",
        "conclusion": "机器学习能够从传感数据中学习设备状态和故障趋势，是预测性维护的重要技术路线。",
        "use": "第一章用它说明数据驱动方法在预测性维护中的重要性，并引出GRU预测模型。",
    },
    {
        "n": 21,
        "source": "Crossref/DOI核查：10.1016/j.jmsy.2022.05.010。",
        "keywords": "RUL预测；机器学习；挑战；综述。",
        "actual": "该文献综述机器学习用于RUL预测的研究现状和挑战。",
        "method": "从数据、特征、模型、泛化能力和评价等方面总结RUL预测问题。",
        "conclusion": "RUL预测的难点在于工业数据噪声、工况变化、模型泛化和实际部署。",
        "use": "第一章用它说明数据驱动RUL方法适合机理难以精确建模但传感数据较丰富的场景。",
    },
    {
        "n": 22,
        "source": "Crossref/DOI核查：10.1016/j.ymssp.2024.111120。",
        "keywords": "物理信息；数据驱动；RUL；综述。",
        "actual": "该文献综述物理信息融合的数据驱动RUL预测研究。",
        "method": "总结如何将物理机理、退化约束或先验知识嵌入数据驱动模型。",
        "conclusion": "融合物理信息可改善纯数据模型的可解释性、鲁棒性和跨工况泛化能力。",
        "use": "第一章用它说明RUL建模存在物理、统计和数据驱动等路线；本文选择较轻量的数据驱动GRU路径。",
    },
    {
        "n": 23,
        "source": "Crossref/DOI核查：10.1016/j.ymssp.2017.11.016。",
        "keywords": "机械健康预测；RUL；数据采集；特征提取。",
        "actual": "该文献系统综述机械健康预测从数据采集到RUL预测的完整流程。",
        "method": "归纳传感数据采集、特征提取、健康指标构造和预测模型等环节。",
        "conclusion": "RUL预测能力依赖有效信号、退化特征、时序建模和复杂工况适应。",
        "use": "第一章用它说明本文RUL预测不仅是回归问题，还依赖时序退化特征提取。",
    },
    {
        "n": 24,
        "source": "Crossref摘要/DOI核查：10.1162/neco.1997.9.8.1735。",
        "keywords": "LSTM；循环神经网络；长期依赖；梯度消失。",
        "actual": "该文献提出LSTM，用于解决普通循环神经网络在长时间间隔信息存储上训练困难的问题。",
        "method": "设计包含门控和记忆单元的循环结构，控制信息写入、保留和输出。",
        "conclusion": "LSTM能够缓解梯度衰减并学习长时依赖。",
        "use": "第一章用该文献作为深度时序模型背景，说明GRU属于门控循环网络发展脉络。",
    },
    {
        "n": 25,
        "source": "Crossref/DOI核查：10.3115/v1/d14-1179。",
        "keywords": "GRU；RNN Encoder-Decoder；序列建模；门控结构。",
        "actual": "该文献提出RNN编码器-解码器并使用GRU结构进行序列表示学习。",
        "method": "通过重置门和更新门控制隐状态更新，使模型能学习序列短语表示。",
        "conclusion": "GRU能够以相对简洁的门控结构完成序列建模。",
        "use": "第一章用该文献作为本文GRU预测器的基础理论来源。",
    },
    {
        "n": 26,
        "source": "arXiv原始页面核查：arXiv:1412.3555；自动元数据匹配结果已剔除。",
        "keywords": "GRU；LSTM；门控循环网络；经验比较。",
        "actual": "该文献经验比较不同门控循环网络在序列建模任务中的表现。",
        "method": "通过实验比较GRU、LSTM和其他循环结构在多个序列任务上的性能。",
        "conclusion": "GRU在若干任务中能够以更简单结构取得与LSTM接近的效果。",
        "use": "第一章用它说明选择GRU具有模型复杂度和时序表达能力之间的平衡。",
    },
    {
        "n": 27,
        "source": "Crossref/DOI核查：10.1016/j.asoc.2023.110419。",
        "keywords": "Attention-GRU；RUL预测；退化序列；预测性维护。",
        "actual": "该文献研究基于注意力GRU的RUL预测，用于从退化序列中提取关键时序信息。",
        "method": "在GRU结构上加入注意力机制，使模型更关注对寿命预测有贡献的时间片段或特征。",
        "conclusion": "注意力机制能够增强GRU对退化信息的表达，提高RUL预测效果。",
        "use": "第一章用它说明GRU及其改进模型已被用于RUL预测，支撑本文采用GRU输出健康信号。",
    },
    {
        "n": 28,
        "source": "Crossref/DOI核查：10.1007/s12206-024-0402-8。",
        "keywords": "LSTM；滚动轴承；RUL；特征构造。",
        "actual": "该文献研究滚动轴承RUL预测，结合特征构造和LSTM模型。",
        "method": "提取C-MMPE等退化特征，并使用LSTM学习轴承退化序列与剩余寿命关系。",
        "conclusion": "深度时序模型结合有效特征可用于具体机械部件的寿命预测。",
        "use": "第一章用该文献说明LSTM类深度序列模型在设备RUL预测中已有应用。",
    },
    {
        "n": 29,
        "source": "Crossref摘要/DOI核查：10.1038/s41598-022-10191-2。",
        "keywords": "涡扇发动机；深度特征提取；特征融合；RUL。",
        "actual": "该文献研究涡扇发动机RUL预测中的噪声、多变量和退化特征表达问题。",
        "method": "采用深度特征提取和融合方法，从发动机传感数据中学习退化表示。",
        "conclusion": "深度特征融合能够改善复杂装备RUL预测中退化趋势表达不足的问题。",
        "use": "第一章用它说明除循环网络外，CNN或深度特征融合也可用于RUL预测。",
    },
    {
        "n": 30,
        "source": "Crossref/DOI核查：10.1016/j.ymssp.2023.110221。",
        "keywords": "注意力网络；并行结构；RUL预测；机械系统。",
        "actual": "该文献研究基于嵌入式注意力并行网络的机器RUL预测。",
        "method": "通过并行网络结构和注意力模块提取多尺度或多通道退化特征。",
        "conclusion": "注意力和并行结构可增强模型对关键退化模式的识别能力。",
        "use": "第一章用它说明RUL预测模型已向更复杂的深度序列结构发展，本文则选择较简洁的GRU用于系统仿真。",
    },
    {
        "n": 31,
        "source": "教材出版信息核查：Pinedo, Scheduling: Theory, Algorithms, and Systems, 5th ed., Springer, 2016。",
        "keywords": "调度理论；作业车间；目标函数；算法。",
        "actual": "该书系统介绍生产调度理论、经典模型、复杂性和算法。",
        "method": "通过数学建模和算法分类讨论单机、并行机、流水车间和作业车间等问题。",
        "conclusion": "调度问题本质上是在有限资源下进行排序和分配，以优化完工、拖期、负载等目标。",
        "use": "第一章用该书作为传统调度理论背景，说明本文问题从经典调度扩展而来。",
    },
    {
        "n": 32,
        "source": "Crossref摘要/DOI核查：10.1287/opre.25.1.45。",
        "keywords": "调度规则；派工规则；优先级；综述。",
        "actual": "该文献综述大量调度规则，说明不同派工规则在排序和调度中的应用。",
        "method": "归纳和分类已有优先规则，并总结它们在仿真研究中的使用。",
        "conclusion": "派工规则计算简单、可解释性强，是车间实时调度的重要基础。",
        "use": "第一章用它说明本文六类调度规则动作空间来自经典派工规则体系。",
    },
    {
        "n": 33,
        "source": "Crossref摘要/DOI核查：10.1287/mnsc.33.8.1035。",
        "keywords": "优先规则；加权拖期；作业车间；交期。",
        "actual": "该文献研究带加权拖期成本的作业车间优先规则。",
        "method": "设计并测试考虑作业权重和交期压力的派工规则。",
        "conclusion": "在存在不同拖期惩罚时，规则需要考虑作业权重和交期紧迫性。",
        "use": "第一章用它说明调度规则可反映拖期压力，但固定规则难以处理维护和健康耦合。",
    },
    {
        "n": 34,
        "source": "Crossref/DOI核查：10.1007/bf02238804。",
        "keywords": "多用途机器；作业车间；机器柔性；调度。",
        "actual": "该文献研究带多用途机器的作业车间调度，是FJSP机器柔性思想的早期基础。",
        "method": "将工序可由多台机器加工的情形纳入作业车间调度建模。",
        "conclusion": "机器选择自由度会显著增加调度问题复杂性。",
        "use": "第一章用它说明FJSP区别于传统JSP的关键在于工序-机器映射不再固定。",
    },
    {
        "n": 35,
        "source": "Crossref/DOI核查：10.1007/bf02023073。",
        "keywords": "柔性作业车间；路径选择；禁忌搜索；调度。",
        "actual": "该文献研究FJSP中的路径选择和调度联合优化。",
        "method": "采用禁忌搜索在机器分配和工序排序空间中搜索解。",
        "conclusion": "元启发式方法能够在大规模组合空间中获得较好调度解。",
        "use": "第一章用它说明早期FJSP求解主要依赖禁忌搜索等元启发式方法。",
    },
    {
        "n": 36,
        "source": "Crossref/DOI核查：10.1002/(sici)1099-1425(200001/02)3:1<3::aid-jos32>3.0.co;2-y。",
        "keywords": "FJSP；邻域函数；局部搜索；组合优化。",
        "actual": "该文献研究FJSP有效邻域函数设计。",
        "method": "通过构造适合FJSP的邻域结构，提高局部搜索在机器选择和排序空间中的效率。",
        "conclusion": "邻域设计对FJSP元启发式求解质量具有关键影响。",
        "use": "第一章用它补充说明静态FJSP中局部搜索类方法的代表性。",
    },
    {
        "n": 37,
        "source": "Crossref/DOI核查：10.1109/tsmcc.2002.1009117。",
        "keywords": "FJSP；多目标优化；进化算法；负载平衡。",
        "actual": "该文献研究柔性作业车间多目标优化问题。",
        "method": "采用定位方法和多目标进化优化，同时考虑多个调度目标。",
        "conclusion": "FJSP不仅是排序问题，还涉及多个目标之间的折中。",
        "use": "第一章用它说明FJSP的多目标性质，为本文THDQN高层目标选择埋下背景。",
    },
    {
        "n": 38,
        "source": "Crossref摘要/DOI核查：10.1007/s40747-024-01374-7。",
        "keywords": "FJSP；并行批处理机器；多种群遗传算法；优化。",
        "actual": "该文献研究带并行批处理机器的FJSP扩展问题。",
        "method": "提出增强型多种群遗传算法解决批处理机器条件下的调度优化。",
        "conclusion": "复杂机器环境下的FJSP需要改进启发式算法处理更大的组合空间。",
        "use": "第一章用它说明近年FJSP仍在通过算法改进扩展问题场景。",
    },
    {
        "n": 39,
        "source": "Crossref/DOI核查：10.1016/j.ejor.2023.05.017。",
        "keywords": "FJSP综述；机器选择；工序排序；研究趋势。",
        "actual": "该文献综述柔性作业车间调度问题的模型、约束和求解方法。",
        "method": "系统整理FJSP研究发展、问题变体、求解算法和未来方向。",
        "conclusion": "FJSP是高度复杂的NP-hard问题，核心难点在于机器选择与工序排序耦合。",
        "use": "第一章用它说明本文调度问题属于FJSP扩展，并具有组合复杂性。",
    },
    {
        "n": 40,
        "source": "Crossref/DOI核查：10.1016/j.cie.2022.108786。",
        "keywords": "集成FJSP；运输；维护；能耗；综述。",
        "actual": "该文献综述集成柔性作业车间调度问题，关注FJSP与其他生产环节的联合建模。",
        "method": "按集成对象和求解方法分类总结FJSP扩展研究。",
        "conclusion": "真实生产中的FJSP常与维护、运输、能耗等约束耦合，不能只看标准模型。",
        "use": "第一章用它说明本文问题不是标准FJSP，而是健康和维护耦合的集成调度问题。",
    },
    {
        "n": 41,
        "source": "OpenAlex摘要/Crossref DOI核查：10.1007/s10845-025-02585-6。",
        "keywords": "动态作业车间；强化学习；实时调度；综述。",
        "actual": "该文献综述强化学习在动态作业车间调度中的应用，强调突发作业、设备故障和需求波动等动态因素。",
        "method": "系统归纳AI驱动动态调度方法、状态动作设计、奖励设置和应用挑战。",
        "conclusion": "强化学习适合处理动态调度中的实时适应问题，但仍面临泛化、可解释性和工业部署挑战。",
        "use": "第一章用它说明本文采用学习型调度器处理事件驱动动态环境具有研究依据。",
    },
    {
        "n": 42,
        "source": "Crossref/DOI核查：10.1109/tcyb.2022.3151855。",
        "keywords": "重调度；新作业插入；预防性维护；多目标FJSP。",
        "actual": "该文献研究新作业插入和机器预防性维护同时存在时的FJSP重调度。",
        "method": "建立多目标重调度模型并考虑新作业和维护事件对原排程的扰动。",
        "conclusion": "动态作业到达和机器维护会共同破坏原有排程，因此需要重调度机制。",
        "use": "第一章用它说明本文事件驱动调度环境中作业到达和维护插入必须共同考虑。",
    },
    {
        "n": 43,
        "source": "Crossref/DOI核查：10.1016/j.jmsy.2021.09.018。",
        "keywords": "生产调度；维护计划；机器退化；状态维护；FJSP。",
        "actual": "该文献研究带机器退化和状态维护的柔性作业车间实时生产调度与维护计划集成。",
        "method": "将机器退化、状态维护和生产调度共同建模，实时协调生产与维护安排。",
        "conclusion": "机器退化会改变生产调度可行性，维护计划应与生产调度同步优化。",
        "use": "第一章用它支撑本文核心观点：维护不是调度后的补丁，而是影响机器可用性的决策动作。",
    },
    {
        "n": 44,
        "source": "Crossref/DOI核查：10.1016/j.cor.2023.106365。",
        "keywords": "多智能体系统；集成调度；维护计划；FJSP。",
        "actual": "该文献研究柔性作业车间中生产调度和维护计划的多智能体集成方法。",
        "method": "构建多智能体系统协调不同决策单元，实现调度和维护计划协同。",
        "conclusion": "多智能体结构有助于表达生产系统中多个相互影响的决策模块。",
        "use": "第一章用它说明本文维护器和调度器拆分为不同智能体具有方法背景。",
    },
    {
        "n": 45,
        "source": "Crossref/DOI核查：10.1016/j.ress.2024.110129。",
        "keywords": "预防性维护；任务调度；可修复多单元系统；集成框架。",
        "actual": "该文献研究可修复多单元系统中预防性维护和任务调度的集成安排。",
        "method": "建立统一框架协调维护任务和生产/系统任务的时间安排。",
        "conclusion": "维护活动会占用系统资源并影响任务执行，因此需要与任务调度联合考虑。",
        "use": "第一章用它说明维护活动会改变资源可用性，必须纳入生产调度模型。",
    },
    {
        "n": 46,
        "source": "Crossref/DOI核查：10.1016/j.jclepro.2025.145856。",
        "keywords": "生产调度；预防性维护；节能；柔性制造；集成优化。",
        "actual": "该文献研究柔性制造环境下生产调度、预防性维护和节能的集成协调。",
        "method": "构建多目标协调框架，将生产、维护和能耗目标统一优化。",
        "conclusion": "真实制造系统中的调度决策往往同时影响维护成本、生产效率和能源表现。",
        "use": "第一章用它说明生产维护协同可扩展到多目标系统优化，本文聚焦拖期、维护和健康状态。",
    },
    {
        "n": 47,
        "source": "Crossref/DOI核查：10.1016/s0004-3702(99)00052-1。",
        "keywords": "时间抽象；Options；MDP；Semi-MDP；层级强化学习。",
        "actual": "该文献提出MDP与半MDP之间的时间抽象框架，是层级强化学习的重要基础。",
        "method": "提出options框架，将持续一段时间的宏动作或子策略嵌入强化学习过程。",
        "conclusion": "复杂决策可通过时间抽象和层级策略分解降低学习难度。",
        "use": "第一章用它说明本文维护端GATE/TYPE和调度端高层目标/低层规则分层设计的理论动机。",
    },
    {
        "n": 48,
        "source": "Crossref/DOI核查：10.1023/a:1022676722315。",
        "keywords": "Q-learning；动作价值；强化学习；贝尔曼更新。",
        "actual": "该文献提出Q-learning，是无模型强化学习中的经典动作价值学习方法。",
        "method": "通过采样状态转移并使用贝尔曼最优方程更新Q值，学习最优动作价值函数。",
        "conclusion": "在满足一定条件时，Q-learning可收敛到最优动作价值函数。",
        "use": "第一章用它说明DQN、DDQN和THDQN都源于动作价值学习思想。",
    },
    {
        "n": 49,
        "source": "Crossref摘要/DOI核查：10.1609/aaai.v30i1.10295。",
        "keywords": "Double DQN；价值过估计；深度强化学习；Q-learning。",
        "actual": "该文献研究DQN中动作价值过估计问题，并提出Double DQN改进。",
        "method": "将动作选择和动作价值评估解耦，使用当前网络选动作、目标网络评估价值。",
        "conclusion": "Double DQN能够降低Q值过估计并提升深度强化学习训练稳定性。",
        "use": "第一章用它说明本文DDQN维护器和THDQN调度器采用Double DQN更新思想。",
    },
]


def p(text: str, style: str = "Normal", bold: bool = False, center: bool = False) -> str:
    jc = '<w:jc w:val="center"/>' if center else ""
    pstyle = f'<w:pStyle w:val="{style}"/>' if style else ""
    rpr = "<w:b/>" if bold else ""
    return (
        "<w:p><w:pPr>"
        f"{pstyle}{jc}<w:spacing w:line=\"360\" w:lineRule=\"auto\" w:after=\"100\"/>"
        "</w:pPr><w:r><w:rPr>"
        f"{rpr}<w:rFonts w:ascii=\"SimSun\" w:eastAsia=\"宋体\" w:hAnsi=\"SimSun\"/><w:sz w:val=\"24\"/>"
        "</w:rPr><w:t xml:space=\"preserve\">"
        f"{escape(text)}"
        "</w:t></w:r></w:p>"
    )


def build_document_xml() -> str:
    body = [
        p("第一章参考文献逐篇核查整理", style="Title", bold=True, center=True),
        p("整理依据：读取《本科毕业设计 傅懋杰 2.0.docx》的第一章和参考文献表，并结合Crossref、OpenAlex、DOI/arXiv/数据集页面等可检索信息核查。无法通过自动元数据正确匹配的报告、教材、arXiv和数据集条目已按原始出处手动标注。"),
        p("说明：以下每条按论文参考文献编号排列，包含核查来源、摘要/关键词、原文实际研究内容、方法或观点、主要结论，以及它在第一章中的引用作用。", bold=True),
    ]
    for e in ENTRIES:
        body.append(p(f"文献[{e['n']}]", style="Heading2", bold=True))
        body.append(p(f"核查来源：{e['source']}"))
        body.append(p(f"关键词/主题：{e['keywords']}"))
        body.append(p(f"原文实际研究内容：{e['actual']}"))
        body.append(p(f"方法或观点：{e['method']}"))
        body.append(p(f"主要结论：{e['conclusion']}"))
        body.append(p(f"第一章如何引用：{e['use']}"))
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:body>" + "".join(body) + sect + "</w:body></w:document>"
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
    <w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto" w:after="100"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:line="360" w:lineRule="auto" w:after="100"/></w:pPr><w:rPr><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:pPr><w:spacing w:before="180" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
</w:styles>
"""


def core_xml() -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>第一章参考文献逐篇核查整理</dc:title>
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
