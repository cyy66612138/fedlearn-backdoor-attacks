import json
import os
import pandas as pd
from openpyxl.styles import Font, Alignment
from typing import List, Dict

# ===================== 通用配置区域（按需修改） =====================
JSON_DIR = "./results"  # JSON文件根目录（支持子目录）
OUTPUT_EXCEL = "federated_learning_experiments.xlsx"  # 输出Excel文件名
TARGET_ROUNDS = [100]  # 需要提取的训练轮数
# 定义需要提取的指标（字段名: Excel列名）
METRICS = {
    "test_accuracy": "测试准确率（%）",
    "test_loss": "测试损失",
    "backdoor_accuracy_0": "后门准确率（%）",
    "backdoor_loss_0": "后门损失"
}
# 百分比指标（需要×100）
PERCENTAGE_METRICS = ["test_accuracy", "backdoor_accuracy_0"]


# =======================================================================

def parse_experiment_config(filename: str) -> Dict[str, str]:
    """
    通用文件名解析函数，从文件名提取攻击类型、防御方法
    适配文件名格式：{attack}_{dataset}_{model}_..._agg_{defense}_opt_{optimizer}.json
    """
    parts = filename.split("_")
    config = {}

    # 1. 提取攻击类型（文件名第一部分）
    config["攻击类型"] = parts[0]

    # 2. 提取防御/聚合方法（在"_agg_"和"_opt_"之间）
    try:
        agg_idx = parts.index("agg")
        opt_idx = parts.index("opt")
        config["防御/聚合方法"] = "_".join(parts[agg_idx + 1:opt_idx])
    except (ValueError, IndexError):
        config["防御/聚合方法"] = "未知方法"

    # 3. 可选：提取数据集、模型（如需补充可扩展）
    # config["数据集"] = parts[1]
    # config["模型"] = parts[2]

    return config


def extract_data_from_json(json_path: str) -> List[Dict]:
    """从单个JSON文件提取目标轮次的所有指标"""
    filename = os.path.basename(json_path)
    experiment_config = parse_experiment_config(filename)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  读取文件失败 {filename}: {str(e)}")
        return []

    # ================== 新增：提取 Alpha 值 ==================
    alpha_val = "N/A"
    config_data = data.get("config", {})

    # 优先从 JSON 内的 config -> dataset -> alpha 中提取
    if "dataset" in config_data and "alpha" in config_data["dataset"]:
        alpha_val = config_data["dataset"]["alpha"]
    else:
        # 如果内部配置中没有，尝试从文件名（如包含 niid_0.5）中提取备份
        parts = filename.split("_")
        if "niid" in parts:
            try:
                niid_idx = parts.index("niid")
                alpha_val = float(parts[niid_idx + 1])
            except (ValueError, IndexError):
                pass
    # =========================================================

    round_data = []
    if "training_history" not in data:
        print(f"⚠️  文件缺少training_history字段: {filename}")
        return []

    # 遍历训练历史，筛选目标轮次
    for record in data["training_history"]:
        rnd = record.get("round")
        if rnd not in TARGET_ROUNDS:
            continue

        # 合并实验配置+轮次+指标，并将 Alpha 写入字典
        row = {
            **experiment_config,
            "Alpha (Non-IID)": alpha_val,
            "训练轮数": rnd
        }

        # 提取所有定义的指标
        for metric_key, metric_col in METRICS.items():
            value = record.get(metric_key)
            if value is None:
                row[metric_col] = None
                continue

            # 百分比指标×100并保留2位小数
            if metric_key in PERCENTAGE_METRICS:
                row[metric_col] = round(value * 100, 2)
            else:
                row[metric_col] = round(value, 6)

        round_data.append(row)

    return round_data


def main():
    # 1. 递归遍历目录，收集所有JSON文件
    all_data = []
    for root, _, files in os.walk(JSON_DIR):
        for filename in files:
            if filename.endswith(".json"):
                json_path = os.path.join(root, filename)
                print(f"正在处理: {os.path.relpath(json_path, JSON_DIR)}")
                all_data.extend(extract_data_from_json(json_path))

    if not all_data:
        print("❌ 未找到任何符合条件的JSON数据，请检查路径配置！")
        return

    # 2. 转换为DataFrame并排序（按攻击类型→Alpha→防御方法→轮数）
    df = pd.DataFrame(all_data)
    sort_columns = ["攻击类型", "Alpha (Non-IID)", "防御/聚合方法", "训练轮数"]
    df = df.sort_values(by=sort_columns).reset_index(drop=True)

    # 3. 重新排列列顺序（把配置列放前面）
    column_order = sort_columns + [col for col in df.columns if col not in sort_columns]
    df = df[column_order]

    # 4. 导出到Excel并美化格式
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="实验数据汇总")

        # 获取工作表对象
        worksheet = writer.sheets["实验数据汇总"]

        # 标题行格式：加粗、居中、字号11
        header_font = Font(bold=True, size=11)
        header_alignment = Alignment(horizontal="center", vertical="center")
        for cell in worksheet[1]:
            cell.font = header_font
            cell.alignment = header_alignment

        # 自适应列宽（最大限制25）
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 25)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"\n✅ 处理完成！共提取 {len(all_data)} 条数据")
    print(f"📊 Excel文件已保存至: {os.path.abspath(OUTPUT_EXCEL)}")


if __name__ == "__main__":
    main()