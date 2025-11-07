import subprocess
import os
import re

def show_tree(root_dir=None):
    """生成项目目录结构"""
    if not root_dir:
        root_dir = os.getcwd()
    
    # 使用tree命令生成目录结构，忽略.gitignore中的文件和target目录
    result = subprocess.run(
        ["tree", "--gitignore", "-I", "target|__pycache__"],
        cwd=root_dir,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

def update_tree_in_readme():
    """更新README.md中的文件结构部分"""
    # 获取新的目录结构
    tree_updated = show_tree()
    
    # 读取README.md文件
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    # 使用正则表达式替换<!-- INSERT BEGIN -->和<!-- INSERT LAST -->之间的内容
    pattern = r"(<!-- INSERT BEGIN -->\n```text\n)(.*?)(\n```\n<!-- INSERT LAST -->)"
    replacement = rf"\g<1>{tree_updated}\g<3>"
    
    # 更新README.md内容
    updated_readme = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)
    
    # 写入更新后的内容
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_readme)
    

if __name__ == "__main__":
    update_tree_in_readme()