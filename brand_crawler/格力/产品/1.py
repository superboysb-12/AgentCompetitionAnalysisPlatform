#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立脚本：格力(Gree)产品JSON数据修复工具 (v2)

功能:
1. 递归扫描指定目录下的所有子文件夹。
2. 查找所有名为 'product_data.json' (或所有 .json) 的文件。
3. 读取每个文件，并从 'description' 字段中提取第一行作为 'product_name'。
4. 将 'description' 字段更新为原始描述中除第一行之外的剩余内容。
5. 将修改后的数据写回原始 .json 文件。

*** 警告: 此脚本会直接修改原始文件，请在运行前备份您的数据。 ***
"""

import json
from pathlib import Path
import os

# --- 配置 ---
# 请修改为您格力挂式空调JSON文件的 *父* 目录
TARGET_DIRECTORY = r"F:\aaaPyCharmprojects\brand_crawler\格力\产品\kongtiao\gree_tezhong"

# 您可以指定只查找 'product_data.json'，或者查找所有 '.json'
# 建议使用 "product_data.json" 更安全
FILE_PATTERN = "product_data.json"


# FILE_PATTERN = "*.json" # 如果您的文件名不固定，请使用这个
# ----------------

def fix_product_names_in_directory(target_dir_str: str):
    """
    递归扫描目录中的JSON文件，并根据 'description' 的第一行
    来修正 'product_name' 和 'description' 字段。
    """

    target_dir = Path(target_dir_str)

    if not target_dir.is_dir():
        print(f"❌ 错误: 目录不存在: {target_dir_str}")
        print("请检查 TARGET_DIRECTORY 变量中的路径是否正确。")
        return

    print(f"🚀 开始递归扫描目录: {target_dir}")

    # --- MODIFICATION START ---
    # 使用 .rglob() 进行递归搜索 (Recursive Glob)
    # 这会查找 target_dir 及其所有子目录中的文件
    print(f"🔍 正在查找所有 '{FILE_PATTERN}' 文件...")
    json_files = list(target_dir.rglob(FILE_PATTERN))
    # --- MODIFICATION END ---

    if not json_files:
        print(f"⚠️ 警告: 在 {target_dir} 及其子目录中未找到任何 '{FILE_PATTERN}' 文件。")
        return

    print(f"🔍 找到了 {len(json_files)} 个文件。开始处理...")

    files_updated = 0
    files_skipped = 0
    files_failed = 0

    for json_file_path in json_files:
        try:
            # 1. 读取JSON文件内容
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取相对路径以便清晰显示
            relative_path = json_file_path.relative_to(target_dir)

            old_product_name = data.get('product_name')
            description = data.get('description')

            # 2. 检查 'description' 是否有效且包含换行符
            if description and '\n' in description:

                # 3. 拆分 'description'
                parts = description.split('\n', 1)
                new_product_name = parts[0].strip()  # 第一行作为新名称
                new_description = parts[1].strip()  # 剩余部分作为新描述

                # 4. 检查是否需要更新
                if (old_product_name != new_product_name) or (data.get('description') != new_description):

                    # 5. 更新数据
                    data['product_name'] = new_product_name
                    data['description'] = new_description

                    # 6. 写回JSON文件 (使用 indent=4 保持格式美观)
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    print(f"✅ [已更新] {relative_path}: '{old_product_name}' -> '{new_product_name}'")
                    files_updated += 1
                else:
                    print(f"ℹ️ [跳过] {relative_path}: 'product_name' 已经正确。")
                    files_skipped += 1

            else:
                print(f"⚠️ [警告] {relative_path}: 'description' 字段为空, 或不包含换行符'\\n'。")
                files_skipped += 1

        except json.JSONDecodeError:
            print(f"❌ [错误] {relative_path}: JSON 格式错误，无法解析。")
            files_failed += 1
        except Exception as e:
            print(f"❌ [严重错误] {relative_path}: {e}")
            files_failed += 1

    # --- 打印总结报告 ---
    print("\n" + "=" * 30)
    print("🎉 处理完成！")
    print(f"   - {files_updated} 个文件被成功更新。")
    print(f"   - {files_skipped} 个文件被跳过（无需更改或有警告）。")
    print(f"   - {files_failed} 个文件处理失败。")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    print("--- 格力JSON数据修复工具 (v2 - 递归版) ---")
    print(f"将要处理的目录: {TARGET_DIRECTORY}")
    print(f"将要查找的文件: {FILE_PATTERN}")
    print("*** 警告: 此操作将直接修改原始文件！***")

    try:
        user_input = input("是否继续? (y/n): ")
    except EOFError:
        user_input = "n"

    if user_input.lower().strip() == 'y':
        fix_product_names_in_directory(TARGET_DIRECTORY)
    else:
        print("操作已取消。")