import asyncio
from async_relation_extractor import AsyncRelationExtractor, extract_relations_async, create_extractor
from config_loader import list_available_configs, list_available_schemas

async def test_config_info():
    """配置信息测试"""
    print("=== 可用配置信息 ===")
    print(f"可用的schema: {list_available_schemas()}")
    print(f"可用的配置: {list_available_configs()}")

async def test_basic_extraction():
    """基础提取测试"""
    print("\n=== 基础提取测试 ===")
    text = """2025年4月27日，2025年中国制冷展在上海隆重开幕。海信中央空调作为行业领军品牌，首日以一场AI低碳矩阵发布会，展示了其在绿色低碳与智能化领域的最新成果，吸引了众多行业专家、领导及媒体朋友的关注。

海信集团副总裁、海信家电集团总裁、海信日立总裁胡剑涌先生发表致辞。他表示："全球绿色低碳转型已步入关键加速期，海信中央空调始终以技术创新与品质提升为核心，推动暖通行业向高效、智能、可持续方向迈进。"""

    # 使用默认配置
    result = await extract_relations_async(text)
    print(f"使用默认配置提取到 {len(result)} 个结果")

    # 使用商业分析配置
    result2 = await extract_relations_async(text, config_name="business_analysis")
    print(f"使用商业分析配置提取到 {len(result2)} 个结果")

    if result:
        print(f"\n示例结果:")
        print(f"  范围: {result[0]['range']}")
        print(f"  文本块: {result[0]['chunk'][:100]}...")
        print(f"  提取结果: {result[0]['result']}")


async def test_context_manager():
    """上下文管理器测试"""
    print("\n=== 上下文管理器测试 ===")
    text = "胡剑涌是海信集团副总裁，他在2025年中国制冷展上发表了重要讲话。"

    # 使用高性能配置
    async with create_extractor("high_performance") as extractor:
        result = await extractor.extract_async(text)
        print(f"使用高性能配置提取到 {len(result)} 个结果")

        # 健康检查
        health = await extractor.health_check()
        print(f"健康检查: {'通过' if health else '失败'}")


async def test_batch_extraction():
    """批量提取测试"""
    print("\n=== 批量提取测试 ===")
    texts = [
        "胡剑涌是海信集团副总裁，负责公司战略规划。",
        "海信中央空调发布了全新的G3系列产品，具有节能环保的特点。",
        "2025年中国制冷展将于4月27日在上海举办，主办方是中国制冷协会。"
    ]

    # 使用内存优化配置进行批量处理
    async with create_extractor("memory_optimized") as extractor:
        batch_results = await extractor.batch_extract_async(texts)
        print(f"使用内存优化配置批量处理 {len(texts)} 个文本")

        for i, results in enumerate(batch_results):
            print(f"\n文本 {i+1} 结果数量: {len(results)}")
            if results:
                print(f"  示例结果: {results[0]['result']}")


async def test_custom_schema():
    """自定义schema测试"""
    print("\n=== 自定义schema测试 ===")
    custom_schema = {
        "公司": ["名称", "业务范围", "产品"],
        "会议": ["名称", "时间", "地点", "议题"]
    }

    text = "腾讯公司是一家专注于互联网服务的科技企业，主要产品包括微信和QQ。"

    # 使用自定义schema
    async with create_extractor("standard", schema=custom_schema) as extractor:
        result = await extractor.extract_async(text)
        print(f"使用自定义schema提取到 {len(result)} 个结果")

        for item in result:
            print(f"  结果: {item['result']}")

    # 使用schema_key
    result2 = await extract_relations_async(text, schema_key="technology")
    print(f"\n使用技术schema提取到 {len(result2)} 个结果")


async def test_performance():
    """性能测试"""
    print("\n=== 性能测试 ===")
    import time

    # 生成较长的测试文本
    base_text = """海信集团副总裁胡剑涌在2025年中国制冷展上发表重要讲话。他表示，海信中央空调将继续致力于技术创新，推出更多节能环保的产品。本次展会展示了G3系列商用多联机等多款新产品，具有高效节能、智能控制等技术亮点。"""

    long_text = base_text * 10  # 重复10次制造较长文本

    start_time = time.time()

    # 使用高性能配置进行测试
    async with create_extractor("high_performance", batch_size=16) as extractor:
        result = await extractor.extract_async(long_text)

    end_time = time.time()

    print(f"处理文本长度: {len(long_text)} 字符")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    print(f"提取结果数量: {len(result)}")


async def test_parameter_override():
    """参数覆盖测试"""
    print("\n=== 参数覆盖测试 ===")
    text = "海信中央空调推出了创新的节能技术。"

    # 使用参数覆盖
    async with create_extractor(
        "standard",
        batch_size=4,
        chunk_max=400
    ) as extractor:
        result = await extractor.extract_async(text)
        print(f"使用参数覆盖提取到 {len(result)} 个结果")


async def main():
    """运行所有测试"""
    try:
        await test_config_info()
        await test_basic_extraction()
        await test_context_manager()
        await test_batch_extraction()
        await test_custom_schema()
        await test_parameter_override()
        await test_performance()

        print("\n=== 所有测试完成 ===")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())