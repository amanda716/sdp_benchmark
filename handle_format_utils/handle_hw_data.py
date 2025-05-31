import re
from pathlib import Path


"""
    用于修改sdp网站下载的hw_data文件名的脚本
    home: 1008 record length
    office: 268 record length
"""
# 替换为实际路径pattern = r'^csi_\d{4}_\d{2}_\d{2}'  # 匹配 "csi_YYYY_MM_DD" 格式
folder_path = Path(r"F:\Repository\pyProjects\sdp_benchmark\data\hw_data_office")
pattern = r'^csi_\d{4}_\d{2}_\d{2}'  # 匹配 "csi_YYYY_MM_DD" 格式
replaced_text = 'csi_268_'

for txt_file in folder_path.rglob('*.txt'):
    if re.match(pattern, txt_file.stem, re.IGNORECASE):
        # 在 "csi_" 后插入 replace_text
        new_stem = txt_file.stem.replace('csi_', replaced_text, 1)
        new_path = txt_file.with_name(f"{new_stem}{txt_file.suffix}")

        txt_file.rename(new_path)
        print(f"已重命名: {txt_file} → {new_path}")
