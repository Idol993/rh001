"""
交互式 GitHub 提交脚本
在终端运行时安全输入 Token
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from push_to_github import push_to_github, collect_files
import getpass


def main():
    print()
    print("=" * 60)
    print("   GitHub 代码提交工具 (无需Git客户端)")
    print("=" * 60)
    print()

    repo = "Idol993/rh001"
    default_msg = "round-1: 提交代码 - 智能炒股分析APP核心逻辑 (多因子特征工程 + 时序预测模型 + 跨端代码生成)"

    print(f"目标仓库: {repo}")
    print()

    token = getpass.getpass("请粘贴你的 GitHub Token (输入时无显示，粘贴后回车): ")
    token = token.strip()

    if not token:
        print("[错误] Token 不能为空")
        return 1

    print()
    msg_input = input(f"请输入提交信息 (直接回车使用默认): ").strip()
    message = msg_input if msg_input else default_msg
    print()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    success = push_to_github(token, repo, message, project_root)

    print()
    print("按回车键退出...")
    try:
        input()
    except:
        pass

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
