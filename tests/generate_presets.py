"""Generate variable_presets.xlsx for the ProScore test dataset.

Usage::

    pip install openpyxl
    python tests/generate_presets.py
"""

import pandas as pd


def main() -> None:
    data = [
        {"variable": "income", "name_cn": "年收入", "dimension": "还款能力",
         "monotonic": "decreasing", "special_values": "-999"},
        {"variable": "debt_ratio", "name_cn": "负债收入比", "dimension": "负债水平",
         "monotonic": "increasing", "special_values": ""},
        {"variable": "age", "name_cn": "年龄", "dimension": "个人信息",
         "monotonic": "u", "special_values": ""},
        {"variable": "utilization", "name_cn": "授信使用率", "dimension": "负债水平",
         "monotonic": "increasing", "special_values": ""},
        {"variable": "credit_months", "name_cn": "信用历史月数", "dimension": "信用历史",
         "monotonic": "decreasing", "special_values": ""},
        {"variable": "num_inquiries", "name_cn": "近期查询次数", "dimension": "查询行为",
         "monotonic": "increasing", "special_values": ""},
        {"variable": "education", "name_cn": "学历", "dimension": "个人信息",
         "monotonic": "", "special_values": ""},
        {"variable": "employment_type", "name_cn": "就业类型", "dimension": "个人信息",
         "monotonic": "", "special_values": ""},
        {"variable": "home_ownership", "name_cn": "房产状态", "dimension": "资产状况",
         "monotonic": "", "special_values": ""},
        {"variable": "loan_purpose", "name_cn": "贷款用途", "dimension": "贷款特征",
         "monotonic": "", "special_values": ""},
    ]

    df_vars = pd.DataFrame(data)
    df_dims = pd.DataFrame([
        {"dimension": "还款能力", "name_cn": "还款能力", "description": "收入类指标，评估借款人偿还能力"},
        {"dimension": "负债水平", "name_cn": "负债水平", "description": "负债类指标，评估当前债务负担"},
        {"dimension": "个人信息", "name_cn": "个人信息", "description": "人口统计特征，用于风险画像"},
        {"dimension": "信用历史", "name_cn": "信用历史", "description": "信用记录长度，反映信用积累"},
        {"dimension": "查询行为", "name_cn": "查询行为", "description": "近期信用查询，反映资金需求紧迫度"},
        {"dimension": "资产状况", "name_cn": "资产状况", "description": "资产所有权，反映财富积累"},
        {"dimension": "贷款特征", "name_cn": "贷款特征", "description": "贷款自身属性"},
    ])

    with pd.ExcelWriter("tests/variable_presets.xlsx", engine="openpyxl") as writer:
        df_vars.to_excel(writer, sheet_name="variables", index=False)
        df_dims.to_excel(writer, sheet_name="dimensions", index=False)

    print("File created: tests/variable_presets.xlsx")
    print(f"\nvariables sheet ({len(df_vars)} rows):")
    print(df_vars.to_string())
    print(f"\ndimensions sheet ({len(df_dims)} rows):")
    print(df_dims.to_string())


if __name__ == "__main__":
    main()
