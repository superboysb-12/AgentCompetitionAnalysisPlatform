import re
from paddlenlp import Taskflow
from typing import List, Tuple

# 你自己的抽取目标
schema = {
    "人物": ["头衔", "所属机构", "观点"],
    "产品": ["型号", "技术亮点", "应用场景"],
    "活动": ["时间", "地点", "主办方"]
}

CHUNK_MIN = 300   # 每块最小字数（中文字符数）
CHUNK_MAX = 800   # 每块最大字数
STRIDE_RATIO = 0.25  # 滑窗重叠比例（25%）

def normalize_text(s: str) -> str:
    #用于标准化换行
    s = re.sub(r'\r', '\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def dedup_paragraphs(paras: List[str]) -> List[str]:
    seen = set()
    kept = []
    for p in paras:
        key = re.sub(r'\s+', '', p)
        if len(key) < 20:
            kept.append(p); continue
        # 粗糙去重：截断做指纹
        fp = key[:80]
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(p)
    return kept

def split_sentences(para: str) -> List[str]:
    # 句号/问号/叹号/分号 + 中英文
    parts = re.split(r'(?<=[。！？；.!?])\s*', para)
    return [p for p in parts if p and p.strip()]

def make_chunks(sentences: List[str]) -> List[Tuple[str, Tuple[int,int]]]:
    """
    返回 [(chunk_text, (sent_start_idx, sent_end_idx)), ...]
    记录句子级起止索引，便于回映射/去重。
    """
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        j = i + 1
        while j < len(sentences) and len(chunk) < CHUNK_MAX:
            next_s = sentences[j]
            if len(chunk) < CHUNK_MIN or len(chunk) + len(next_s) <= CHUNK_MAX:
                chunk += next_s
                j += 1
            else:
                break
        chunks.append((chunk, (i, j-1)))
        # 滑窗前进（保留重叠）
        step = max(1, int((j - i) * (1 - STRIDE_RATIO)))
        i += step
    return chunks

class RelationExtractor:
    def __init__(self):
        self.ie = Taskflow(
            "information_extraction",
            schema=schema,
            model="paddlenlp/PP-UIE-0.5B",
            schema_lang="zh",
            # 若硬件不支持 bfloat16，可去掉 precision 参数或改为 "fp16"
            batch_size=8 , # 批量预测更快（官方文档推荐批量更快）
            precision="bfloat16"
        )

    def extract(self, text: str):
        text = normalize_text(text)
        paras = [p for p in re.split(r'\n{2,}', text) if p.strip()]
        paras = dedup_paragraphs(paras)

        chunk_infos = []
        for p in paras:
            sents = split_sentences(p)
            if not sents:
                continue
            chunk_infos += make_chunks(sents)

        if not chunk_infos:
            return []

        chunks = [c for c, _ in chunk_infos]

        # 一次批量调用
        raw_results = self.ie(chunks)

        # 简单去重合并（基于文本片段字符串）
        # 你也可以按 (label, span_text, parent_text_idx) 粒度更细地做
        seen = set()
        merged = []
        for (chunk, idx_range), res in zip(chunk_infos, raw_results):
            key = str(res)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"range": idx_range, "chunk": chunk, "result": res})
        return merged

def main():
    extractor = RelationExtractor()
    text = """2025年4月27日，2025年中国制冷展在上海隆重开幕。海信中央空调作为行业领军品牌，首日以一场AI低碳矩阵发布会，展示了其在绿色低碳与智能化领域的最新成果，吸引了众多行业专家、领导及媒体朋友的关注。\n\n年4月27日，2025年中国制冷展在上海隆重开幕。海信中央空调作为行业领军品牌，首日以一场AI低碳矩阵发布会，展示了其在绿色低碳与智能化领域的最新成果，吸引了众多行业专家、领导及媒体朋友的关注。\n\n作为暖通行业的领跑者，海信中央空调凭借深厚的技术积累与对市场需求的敏锐洞察，不断拓展绿色低碳技术的应用边界。此次AI低碳矩阵发布会的举办，是海信中央空调持续探索细分场景、优化解决方案的有力彰显。海信中央空调AI低碳矩阵发布会的现场各行业专家和领导先后进行致辞。\n\n发布会伊始，海信集团副总裁、海信家电集团总裁、海信日立总裁胡剑涌先生发表致辞。他表示：“全球绿色低碳转型已步入关键加速期，海信中央空调始终以技术创新与品质提升为核心，推动暖通行业向高效、智能、可持续方向迈进。”胡总重点介绍了本次展会的三大战略级创新成果：全球首台10kV正压液浮无油变频离心机、ECO-B智慧楼宇与能源管理系统2.0、G3系列商用多联机&G3热擎系列。\n\n胡剑涌强调：海信将以智慧赋能绿色未来为使命，持续推动技术创新与产业升级，为“双碳”目标贡献中国智慧！\n\n全国工程勘察设计大师、中国勘察设计协会建筑环境与能源应用分会名誉会长罗继杰大师从工程设计角度对暖通行业的低碳发展进行了阐述。他指出，建筑节能减碳离不开优秀的设备及解决方案，而海信中央空调的创新产品正是对未来建筑绿色发展趋势的有力响应。罗大师强调：“真正的绿色转型需要技术创新与系统思维的双轮驱动。海信中央空调的这些成果，不仅满足了当下市场需求，更前瞻性地布局了未来可持续发展路径，展现了企业卓越的战略眼光和技术实力。”\n\n全国工程勘察设计大师、中国勘察设计协会建筑环境与能源应用分会名誉会长罗继杰大师\n\n从工程设计角度对暖通行业的低碳发展进行了阐述。他指出，建筑节能减碳离不开优秀的设备及解决方案，而海信中央空调的创新产品正是对未来建筑绿色发展趋势的有力响应。罗大师强调：“真正的绿色转型需要技术创新与系统思维的双轮驱动。海信中央空调的这些成果，不仅满足了当下市场需求，更前瞻性地布局了未来可持续发展路径，展现了企业卓越的战略眼光和技术实力。”\n\n中国制冷空调工业协会会长、全国冷冻空调设备标准化技术委员会主任委员、合肥通用机电产品检测院院长李江院长则从行业发展的高度，分析了暖通空调行业的转型机遇。他提到：“海信中央空调的创新实践正在重新定义暖通产品的价值标准：从关注单机性能转向追求系统能效，从满足基本功能转向提供智慧服务，从产品制造商转型为解决方案企业。这种转变，正是我们这个时代最需要的产业升级。”\n\n中国制冷空调工业协会会长、全国冷冻空调设备标准化技术委员会主任委员、合肥通用机电产品检测院院长李江院长\n\n则从行业发展的高度，分析了暖通空调行业的转型机遇。"""
    result = extractor.extract(text)
    from pprint import pprint
    pprint(result)


if __name__ == "__main__":
    main()