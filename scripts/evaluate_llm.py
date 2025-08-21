import os
import json
import logging
import re
import random
import statistics
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from datetime import datetime

# 创建日志目录
os.makedirs("outputs/logs", exist_ok=True)

# 配置日志到文件
log_filename = f"outputs/logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 创建全局锁，确保线程安全的文件操作
lock = threading.Lock()

class Evaluator:
    """大纲评估器类"""
    
    def __init__(self, judge_api_url: str, judge_api_key: str, judge_model: str, timeout: int = 3600):
        """
        初始化评估器
        
        Args:
            judge_api_url: 评估API的URL
            judge_api_key: 评估API的密钥
            judge_model: 评估模型名称
            timeout: 请求超时时间（秒）
        """
        self.judge_api_url = judge_api_url
        self.judge_api_key = judge_api_key
        self.judge_model = judge_model
        self.timeout = timeout
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=judge_api_key,
            base_url=judge_api_url,
            timeout=timeout
        )
        
        logger.info(f"Evaluator initialized with model: {judge_model}")
        logger.info(f"Log file: {log_filename}")
    
    def process_outline(self, data_item):
        """处理单个大纲评估任务"""
        try:
            outline = data_item.get("generated", "")
            topic = data_item.get("topic", "Unknown Topic")
            item_id = data_item.get("id", "unknown")
            
            # 优化提示词：明确要求JSON格式，示例模板
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "您是一位专业的学术写作评估专家，请专注于分析科研文献综述类论文大纲（大纲指的是文章的所有标题，包含一级标题、二级标题等）的结构与内容质量。"
                    },
                    {
                        "role": "user",
                        "content": (
                            f"请评估以下关于'{topic}'主题的文献综述大纲的质量，并从以下六个维度进行打分。评价过程中请确保对六个方面的评价明确独立，没有重叠：对每一个维度的评价仅考虑该维度本身，严格对照下面的指标描述判断，不掺杂其他的维度干扰。"
                            + "每个维度的评分为 0 到 10 分（可以包含0.5分）。请依据以下标准进行评估：\n\n"
                            + "【结构】\n"
                            + "## 结构 - 信息快速定位：\n"
                            + "- 读者可以快速定位信息，大纲应符合所属领域文献综述的体裁规范，并采用学界公认的常规结构之一，以便于读者对于章节内容有所预期，可以快速定位想查找的信息。常见结构及其适用领域包括：\n"
                            + "  IMRaD 结构（Introduction – Methods – Results – Discussion），典型应用于医学、生命科学和实验型自然科学领域；\n"
                            + "  按时间顺序，即沿事件或成果出现的年代串联文献，突出历史演进脉络，常用于物理学理论演变、技术史、社会科学中概念变迁等领域；\n"
                            + "  按理论发展顺序，依据理论内部推演层次，从基础原理到最新扩展逐级展开，适用于数学、理论计算机科学、哲学等需要突出概念体系及证明链的领域；\n"
                            + "  按主题或方法分类（分类学式结构），通过鉴别并并列介绍领域内主要主题、子领域或方法学流派，常见于计算机科学与工程技术（按算法或技术路线构建 taxonomy）、教育学、心理学、管理学（围绕变量或研究范式如定量、定性、混合方法分章）；\n"
                            + "  按研究问题或子议题划分，以若干预设的关键研究问题为主线，每节对应一个问题并整合回答，适用于系统综述、循证医学、公共政策研究等问题导向型领域。\n"
                            + " - 评审要点是，大纲必须清楚表明自己采用了上述哪一种结构，并保持内部一致，以确保读者能快速定位信息。若大纲明显未遵循任何规范结构，或多种结构混用导致逻辑混乱、读者难以快速检索内容，该维度得分将不超过 3 分。若大纲大体上符合某一类范式，但中间穿插额外段落，额外段落不影响信息定位的高效性，或者若大纲虽未明确采用标准结构范式，但各章节功能划分清晰、内容独立明确，仍能帮助读者迅速定位信息，得分范围为 3 - 6 分。若大纲遵循领域的规范结构，且各章节编排合理、逻辑一致，能有效引导读者理解内容脉络，则得分可达到 6 分以上。除此之外，如果大纲含有功能段（如Acknowledgements、Author Contributions、Data & Code Availability、Appendix），应依照学术惯例合理放置，便于读者查阅（对于不含功能段的大纲一律视为满足要求）。\n\n"
                            + "## 结构 - 详略得当：\n"
                            + "- 大纲需体现鲜明的详略差异，依据各部分研究主题的重要性、深度与复杂度，进行差异化布局：对核心议题，通过更多数量的次级小标题或更深的层级拆分，展开深入探讨；对次要、重复性内容，以精简少量的小标题（或合并阐述）简要带过 。以此实现结构详略得当，突出主线，让读者快速识别核心内容，避免 \"平均用力\" 导致的主题罗列感，保障结构清晰性。\n"
                            + "- 若大纲中各主章节的次级小标题数量、层级深度高度趋同（无明显核心内容聚焦），则该维度赋分3 分以下。\n\n"
                            + "【内容】\n"
                            + "## 内容 - 章节互斥性：\n"
                            + "- 大纲应保证同层级内容互斥原则，各章节/小节覆盖内容应无重复，避免主题交叉混乱，确保读者清晰理解各部分的逻辑分工与内容边界，以便于读者通过大纲快速定位信息位置。\n"
                            + "- 对于存在明显内容上有重叠的大纲，赋分3分以下。若出现轻微的内容交叉情况，虽不致完全混淆逻辑，但仍影响信息定位的清晰度，则赋3-6分。\n\n"
                            + "## 内容 - 逻辑深度：\n"
                            + "- 大纲应采用多样化的论证线索范式，如部分章节按 \"先总体后具体\"\" 因果关系 \"\"对立观点辩论\"\" 时间发展脉络 \"\"理论 - 方法 - 应用\" 或 \"挑战 - 解决 - 评价\" 等逻辑编排。需避免同级标题之间仅呈现并列关系，或者只是列出一个接一个的主题，而缺乏内在联系与逻辑递进，可能暗示逻辑深度不够。\n"
                            + "- 大纲的全部或部分内容应体现深度的逻辑递进链条，由紧密相关的章节构成较长的逻辑链条簇。例如，多个小章节按发展顺序或层层递进关系紧密衔接，以此彰显大纲的逻辑深度。大纲在逻辑复杂或内容丰富的章节引入三级标题，进一步增强逻辑层次。应避免大章节下的所有小章节全部彼此独立，无法形成逻辑依赖关系，导致难以构建深层逻辑链条\n"
                            + "- 如果大纲中的各部分仅存在并列关系，反映出对于主题的罗列而没有递进，则赋分3分以下。如果大纲中具有较为丰富的论证线索范式，但是不具有较长的逻辑链条簇，层次扁平没有深度，赋3-6分。如果大纲中运用丰富的论证线索范式，同时反映出深入的逻辑递进，可以赋分6分以上。在适当部分引入三级标题以增加逻辑层次和可读性者，可以额外加分，优秀者可赋分8分以上。\n\n"
                            + "## 内容 - 学术价值：\n"
                            + "- 大纲应明确列出以下的至少一点学术贡献：\n"
                            + "  单列\"局限与研究空白\"章节，系统梳理尚未解决的问题，为后续工作铺路；大纲上看，如果出现\"挑战与机遇\"\"现有问题\"\"研究空白\"等字样的段落或章节，则表明作者有意识地评价了领域现状；\n"
                            + "  提出新的概念框架、分类法或综合模型，展示对现有知识的再组织与创新贡献；如果大纲中有专门章节如\"概念框架\"\"模型构建\"或在引言/讨论中明确指出将构建一个整合模型，则体现出作者在文献基础上创造性地产出了新理论见解；\n"
                            + "  包含具体且可操作的\"未来研究方向\"或\"研究议程\"，为学界指路；综述都会在结尾对未来研究方向做出展望，这种未来展望有时以\"Future Directions\"独立成节，或者融入结论段落；\n"
                            + "  在应用性强的领域（如医学、社会政策、工程应用等）设置\"实践/政策启示；小节，强调成果转化价值，这部分总结对实践的启示，例如对临床实践、产业应用或政策制定的建议。体现在结构上——有的综述在讨论部分增加\"对实践的意义\"之类的段落，或者在结论中合并谈论实践/政策影响。比如医学综述会写\"对临床的意义和未来研究\"，管理学综述会有\"管理实践启示\"。如果大纲中特别包含了实践建议相关内容，说明作者关注将学术知识转化为应用价值，这也是学术贡献的一种体现。\n"
                            + " - 如果大纲不明确包含以上任意一种学术价值的体现，则赋分3分以下。如果大纲虽提及至少 1 种学术贡献，却未展开细化阐述，如仅在标题中列出 \"未来研究方向\" 或 \"研究空白\"，但缺乏次级标题对其具体内容、影响或实施路径进行说明，导致学术价值未能充分体现，则得分范围为 3 - 6 分。\n\n"
                            + "【语用】\n"
                            + "## 语用 - 描述性与简洁性：\n"
                            + "- 大纲的标题应具有描述性，让读者一眼就明白研究的大致内容，能描绘明确对象，能让读者预判文章/章节涉及的问题域。避免一些宽泛的措辞的标题（例如，Concepts, Definitions,Proofs, Examples, Observations, Discussion...）。对于一级标题可以适当宽容，如果一级标题下方的二级标题体现了具体的描述对象或者主题，也可以视作满足条件，否则视作不满足该标准。\n"
                            + "- 大纲标题在具备描述性的基础上应尽量简洁。标题应避免复杂的语法结构；例如，避免使用过长的名词串或复杂从句，采用直接明了的表达。大纲标题应保持精炼，不超过 8个词（若标题含必要长关键术语则可豁免），剔除冗余字词。 \n"
                            + "-  评分时，如果大纲几乎不具备描述性，从大纲中无法知晓任何有关于主题的具体对象或者信息，则赋3分以下。如果大纲具有描述性，但是用语复杂，存在超过8个词以上的标题，则赋值3-6分。\n\n"
                            + "请严格按照以下 JSON 格式输出结果，不要添加任何额外说明或解释: \n"
                            + "{\n"
                            + "  \"评价\": \"详细具体评分理由，必须结合具体例子阐述对于每个维度打分的理由，明确指出文章大纲中什么部分违反了该维度标准\",\n"
                            + "  \"结构_信息快速定位\": 0-10 的数值（可包含0.5）,\n"
                            + "  \"结构_详略得当\": 0-10 的数值（可包含0.5）,\n"
                            + "  \"内容_章节互斥性\": 0-10 的数值（可包含0.5）,\n"
                            + "  \"内容_逻辑深度\": 0-10 的数值（可包含0.5）,\n"
                            + "  \"内容_学术价值\": 0-10 的数值（可包含0.5）,\n"
                            + "  \"语用_描述性与简洁性\": 0-10 的数值（可包含0.5）\n"
                            + "}\n\n"
                            + "待评估的大纲如下：\n"
                            + f"{outline}"
                        )
                    }
                ]
            )

            # 提取并清理模型响应
            raw_response = response.choices[0].message.content.strip()

            # 移除 Markdown 代码块标记（如果存在）
            raw_response = re.sub(r'```json\s*', '', raw_response)
            raw_response = re.sub(r'\s*```', '', raw_response)

            # 确保 JSON 对象被花括号包裹
            if not raw_response.startswith('{'):
                raw_response = '{' + raw_response
            if not raw_response.endswith('}'):
                raw_response = raw_response + '}'

            # 处理JSON中的换行符问题
            # 将未转义的换行符替换为转义的换行符
            raw_response = re.sub(r'(?<!\\)\n', '\\n', raw_response)
            # 处理可能存在的其他未转义字符
            raw_response = re.sub(r'(?<!\\)\r', '\\r', raw_response)
            raw_response = re.sub(r'(?<!\\)\t', '\\t', raw_response)

            # 尝试解析JSON
            try:
                result = json.loads(raw_response)
            except json.JSONDecodeError as e:
                # 如果仍然解析失败，尝试更激进的清理
                logger.warning(f"First JSON parsing attempt failed for topic '{topic}' (item_id: {item_id}): {e}")
                
                # 策略1: 移除所有换行符和制表符
                raw_response_cleaned = re.sub(r'[\n\r\t]', ' ', raw_response)
                try:
                    result = json.loads(raw_response_cleaned)
                except json.JSONDecodeError as e2:
                    logger.warning(f"Second JSON parsing attempt failed for topic '{topic}' (item_id: {item_id}): {e2}")
                    
                    # 策略2: 处理未转义的引号问题
                    # 找到所有字符串值并修复其中的引号
                    try:
                        # 使用正则表达式提取评分部分
                        scores = {}
                        score_patterns = [
                            r'"结构_信息快速定位":\s*([0-9.]+)',
                            r'"结构_详略得当":\s*([0-9.]+)',
                            r'"内容_章节互斥性":\s*([0-9.]+)',
                            r'"内容_逻辑深度":\s*([0-9.]+)',
                            r'"内容_学术价值":\s*([0-9.]+)',
                            r'"语用_描述性与简洁性":\s*([0-9.]+)'
                        ]
                        
                        score_names = [
                            "结构_信息快速定位",
                            "结构_详略得当", 
                            "内容_章节互斥性",
                            "内容_逻辑深度",
                            "内容_学术价值",
                            "语用_描述性与简洁性"
                        ]
                        
                        for pattern, name in zip(score_patterns, score_names):
                            match = re.search(pattern, raw_response_cleaned)
                            if match:
                                scores[name] = float(match.group(1))
                        
                        if len(scores) >= 4:  # 至少要有4个评分才认为是有效的
                            # 提取评价内容（简化处理）
                            evaluation_match = re.search(r'"评价":\s*"([^"]*(?:"[^"]*"[^"]*)*)"', raw_response_cleaned)
                            evaluation = evaluation_match.group(1) if evaluation_match else "评价内容提取失败"
                            
                            # 构建修复后的结果
                            result = {
                                "评价": evaluation,
                                **scores
                            }
                            logger.info(f"Successfully extracted scores using regex for topic '{topic}' (item_id: {item_id})")
                        else:
                            raise json.JSONDecodeError("Insufficient scores extracted", raw_response_cleaned, 0)
                            
                    except Exception as e3:
                        logger.error(f"All JSON parsing attempts failed for topic '{topic}' (item_id: {item_id}): {e3}")
                        return {
                            "topic": topic,
                            "id": item_id,
                            "error": f"JSON解析失败: {str(e3)}",
                            "raw_response": raw_response,
                            "success": False
                        }
            
            # 处理数组响应 - 如果结果是数组，取第一个元素
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]
                else:
                    return {
                        "topic": topic,
                        "id": item_id,
                        "error": "Empty array response",
                        "raw_response": raw_response,
                        "success": False
                    }
            
            # 提取所有评分项
            scores = {
                "结构_信息快速定位": float(result.get('结构_信息快速定位', 0)),
                "结构_详略得当": float(result.get('结构_详略得当', 0)),
                "内容_章节互斥性": float(result.get('内容_章节互斥性', 0)),
                "内容_逻辑深度": float(result.get('内容_逻辑深度', 0)),
                "内容_学术价值": float(result.get('内容_学术价值', 0)),
                "语用_描述性与简洁性": float(result.get('语用_描述性与简洁性', 0))
            }
            evaluation = result.get('评价', '无评价')

            return {
                "topic": topic,
                "id": item_id,
                "evaluation": evaluation,
                "scores": scores,
                "raw_response": raw_response,
                "success": True
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Model response is not valid JSON for topic '{topic}' (item_id: {item_id})")
            return {
                "topic": topic,
                "id": item_id,
                "error": f"JSON解析失败: {str(e)}",
                "raw_response": raw_response,
                "success": False
            }
        except Exception as e:
            logger.error(f"Evaluation failed for topic '{topic}' (item_id: {item_id}): {str(e)}")
            return {
                "topic": topic,
                "id": item_id,
                "error": str(e),
                "success": False
            }
    
    def evaluate_outline(
        self,
        file_path, 
        max_workers=8, 
        sample_size=None, 
        random_seed=None, 
        use_sampling=True,
        output_file_path=None
    ):
        """
        评估 JSONL 文件中的大纲，使用并发处理
        
        Args:
            file_path: 输入文件路径
            max_workers: 并发工作线程数
            sample_size: 采样大小，None表示使用全部数据
            random_seed: 随机种子，用于可重复的随机采样
            use_sampling: 是否启用采样，False则处理全量数据
            output_file_path: 结果输出文件路径
            
        Returns:
            dict: 评估结果统计
        """
        # 初始化数据结构
        success_count = 0
        fail_count = 0
        total_scores = {
            "结构_信息快速定位": 0,
            "结构_详略得当": 0,
            "内容_章节互斥性": 0,
            "内容_逻辑深度": 0,
            "内容_学术价值": 0,
            "语用_描述性与简洁性": 0
        }
        
        # 存储所有成功评估的分数，用于计算统计指标
        all_scores = {
            "结构_信息快速定位": [],
            "结构_详略得当": [],
            "内容_章节互斥性": [],
            "内容_逻辑深度": [],
            "内容_学术价值": [],
            "语用_描述性与简洁性": []
        }
        
        results = []
        failed_responses = []
        success_responses = []

        # 读取所有数据
        logger.info(f"Reading data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data_items = [json.loads(line.strip()) for line in f if line.strip()]
        
        # 参数化采样逻辑
        total_data_size = len(data_items)
        
        if use_sampling and sample_size is not None:
            if sample_size < total_data_size:
                if random_seed is not None:
                    random.seed(random_seed)
                    logger.info(f"[随机采样] 使用固定随机种子: {random_seed}")
                    
                data_items = random.sample(data_items, sample_size)
                logger.info(f"[随机采样] 从 {total_data_size} 条数据中随机抽取 {sample_size} 条")
            else:
                logger.warning(f"[警告] 采样大小 {sample_size} 大于数据总量 {total_data_size}，使用全部数据")
        else:
            logger.info(f"[全量处理] 处理全部 {total_data_size} 条数据")
        
        total_lines = len(data_items)
        logger.info(f"Total items to process: {total_lines}")

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(self.process_outline, item): item 
                for item in data_items
            }
            
            # 处理完成的任务并显示进度
            with tqdm(total=total_lines, desc="Processing outlines") as pbar:
                for future in as_completed(future_to_item):
                    result = future.result()
                    if result["success"]:
                        success_count += 1
                        success_responses.append({
                            "topic": result["topic"],
                            "id": result["id"],
                            "raw_response": result["raw_response"]
                        })
                        
                        # 累加分数并存储到列表中
                        for key in total_scores:
                            score = result["scores"][key]
                            total_scores[key] += score
                            all_scores[key].append(score)
                        
                        results.append({
                            "topic": result["topic"],
                            "id": result["id"],
                            "evaluation": result["evaluation"],
                            "scores": result["scores"]
                        })
                        
                        logger.info(f"Successfully evaluated item {result['id']}: {result['topic']}")
                    else:
                        fail_count += 1
                        failed_responses.append({
                            "topic": result["topic"],
                            "id": result["id"],
                            "error": result.get("error", "Unknown error"),
                            "raw_response": result.get("raw_response", "No response")
                        })
                        
                        logger.warning(f"Failed to evaluate item {result['id']}: {result.get('error', 'Unknown error')}")
                    
                    pbar.update(1)

        # 计算统计指标
        stats_data = self.calculate_statistics(all_scores, success_count, fail_count)
        
        # 输出统计信息
        logger.info(f"\n=== 评估结果统计 ===")
        logger.info(f"成功解析的条目数: {success_count}")
        logger.info(f"解析失败的条目数: {fail_count}")
        logger.info(f"成功率: {success_count/(success_count+fail_count)*100:.1f}%")
        
        logger.info(f"\n=== 各维度统计指标 ===")
        for dimension, stats in stats_data["dimension_stats"].items():
            logger.info(f"\n{dimension}:")
            logger.info(f"  平均分: {stats['mean']:.2f}")
            logger.info(f"  标准差: {stats['std']:.2f}")
            logger.info(f"  最高分: {stats['max']:.2f}")
            logger.info(f"  最低分: {stats['min']:.2f}")
            logger.info(f"  中位数: {stats['median']:.2f}")

        # 保存结果到文件
        output_data = {
            "results": results,
            "average_scores": stats_data["average_scores"],
            "stats": {
                "successful_entries": success_count,
                "failed_entries": fail_count,
                "total_entries": success_count + fail_count,
                "success_rate": round(success_count/(success_count+fail_count)*100, 2),
                "failed_responses": failed_responses,
                "success_responses": success_responses
            }
        }
        
        if output_file_path is None:
            output_file_path = os.path.join("outputs", "evaluation_results.jsonl")
        
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # 使用锁确保线程安全的文件写入
        with lock:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                for entry in output_data["results"]:
                    outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                outfile.write(json.dumps({"average_scores": stats_data["average_scores"]}, ensure_ascii=False) + "\n")
                outfile.write(json.dumps({"stats": {"successful_entries": success_count, "failed_entries": fail_count}}, ensure_ascii=False) + "\n")
        
        # 保存详细的统计结果到score.json
        score_file_path = os.path.join("outputs", "score.json")
        with open(score_file_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed statistics saved to: {score_file_path}")
        
        # 保存失败的响应到单独的调试文件
        if failed_responses:
            debug_file = output_file_path.replace('.jsonl', '_failed_responses.json')
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(failed_responses, f, ensure_ascii=False, indent=2)
            logger.info(f"Failed responses saved to: {debug_file}")
        
        logger.info(f"Results have been saved to {output_file_path}")
        return output_data
    
    def calculate_statistics(self, all_scores, success_count, fail_count):
        """
        计算详细的统计指标
        
        Args:
            all_scores: 所有成功评估的分数字典
            success_count: 成功数量
            fail_count: 失败数量
            
        Returns:
            dict: 包含所有统计指标的字典
        """
        stats_data = {
            "summary": {
                "total_items": success_count + fail_count,
                "successful_items": success_count,
                "failed_items": fail_count,
                "success_rate": round(success_count/(success_count+fail_count)*100, 2) if (success_count + fail_count) > 0 else 0,
                "evaluation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            "average_scores": {},
            "dimension_stats": {},
            "overall_stats": {}
        }
        
        # 计算各维度的统计指标
        dimension_scores = []
        for dimension, scores in all_scores.items():
            if scores:  # 确保有数据
                mean_score = sum(scores) / len(scores)
                stats_data["average_scores"][dimension] = round(mean_score, 2)
                
                # 计算详细统计指标
                stats_data["dimension_stats"][dimension] = {
                    "count": len(scores),
                    "mean": round(mean_score, 2),
                    "std": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
                    "max": round(max(scores), 2),
                    "min": round(min(scores), 2),
                    "median": round(statistics.median(scores), 2),
                    "q1": round(statistics.quantiles(scores, n=4)[0], 2) if len(scores) > 1 else round(scores[0], 2),
                    "q3": round(statistics.quantiles(scores, n=4)[2], 2) if len(scores) > 1 else round(scores[0], 2)
                }
                
                # 收集所有分数用于总体统计
                dimension_scores.extend(scores)
        
        # 计算总体统计指标
        if dimension_scores:
            stats_data["overall_stats"] = {
                "total_scores": len(dimension_scores),
                "overall_mean": round(sum(dimension_scores) / len(dimension_scores), 2),
                "overall_std": round(statistics.stdev(dimension_scores), 2) if len(dimension_scores) > 1 else 0,
                "overall_max": round(max(dimension_scores), 2),
                "overall_min": round(min(dimension_scores), 2),
                "overall_median": round(statistics.median(dimension_scores), 2)
            }
        
        return stats_data

def main():
    parser = argparse.ArgumentParser(description="评估大纲JSONL文件")
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--judge_api_url', type=str, required=True, help='评估API URL')
    parser.add_argument('--judge_api_key', type=str, required=True, help='评估API密钥')
    parser.add_argument('--judge_model', type=str, required=True, help='评估模型名称')
    parser.add_argument('--max_workers', type=int, default=8, help='并发线程数')
    parser.add_argument('--use_sampling', action='store_true', help='是否启用采样')
    parser.add_argument('--sample_size', type=int, default=None, help='采样数量')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 创建评估器实例
    evaluator = Evaluator(
        judge_api_url=args.judge_api_url,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model
    )

    print(f"Evaluating outlines in {args.input}...")
    evaluator.evaluate_outline(
        args.input,
        max_workers=args.max_workers,
        random_seed=args.random_seed,
        use_sampling=args.use_sampling,
        sample_size=args.sample_size,
        output_file_path=args.output
    )
    print(f"Evaluation completed.")

if __name__ == "__main__":
    main()