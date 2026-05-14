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

    round_data = []
    if "training_history" not in data:
        print(f"⚠️  文件缺少training_history字段: {filename}")
        return []

    # ======== 新增：提取 gamma 和 drop_rate ========
    gamma = "N/A"
    drop_rate = "N/A"

    # 尝试从 config -> client_attacks[0] 中提取超参数
    config_dict = data.get("config", {})
    client_attacks = config_dict.get("client_attacks", [])
    if client_attacks and isinstance(client_attacks, list) and len(client_attacks) > 0:
        first_attack = client_attacks[0]
        if isinstance(first_attack, dict):
            gamma = first_attack.get("gamma", "N/A")
            drop_rate = first_attack.get("drop_rate", "N/A")
    # ===============================================

    # 遍历训练历史，筛选目标轮次
    for record in data["training_history"]:
        rnd = record.get("round")
        if rnd not in TARGET_ROUNDS:
            continue

        # 合并实验配置+轮次+超参数
        row = {
            **experiment_config,
            "训练轮数": rnd,
            "Gamma": gamma,
            "Drop_Rate": drop_rate
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

    # 2. 转换为DataFrame并排序（按攻击类型→防御方法→轮数）
    df = pd.DataFrame(all_data)
    sort_columns = ["攻击类型", "防御/聚合方法", "训练轮数"]
    df = df.sort_values(by=sort_columns).reset_index(drop=True)

    # 3. 重新排列列顺序（把配置列放前面，Gamma和Drop_Rate紧随其后）
    base_columns = sort_columns + ["Gamma", "Drop_Rate"]
    column_order = base_columns + [col for col in df.columns if col not in base_columns]
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