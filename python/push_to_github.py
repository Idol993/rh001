"""
GitHub API 代码提交脚本
======================
不依赖Git客户端，直接通过GitHub REST API提交代码到远程仓库

使用方法:
    python push_to_github.py --token ghp_xxx --repo Idol993/rh001 --message "提交说明"
"""

import urllib.request
import urllib.error
import json
import base64
import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple


class GitHubAPI:
    """GitHub REST API 客户端"""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "SmartTradingApp-Deployer",
        }

    def _request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Tuple[int, dict]:
        """发送HTTP请求"""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(url, data=body, method=method)
        for k, v in self.headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req) as resp:
                resp_body = resp.read().decode("utf-8")
                return resp.status, json.loads(resp_body) if resp_body else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
            except:
                err_json = {"error": err_body}
            return e.code, err_json

    def check_repo_exists(self) -> bool:
        """检查仓库是否存在"""
        status, data = self._request("GET", f"repos/{self.repo}")
        return status == 200

    def get_default_branch(self) -> Optional[str]:
        """获取仓库默认分支"""
        status, data = self._request("GET", f"repos/{self.repo}")
        if status == 200:
            return data.get("default_branch", "main")
        return None

    def get_latest_commit(self, branch: str) -> Optional[dict]:
        """获取分支最新commit信息"""
        status, data = self._request("GET", f"repos/{self.repo}/git/ref/heads/{branch}")
        if status == 200:
            return data
        return None

    def get_commit(self, commit_sha: str) -> Optional[dict]:
        """获取commit详情"""
        status, data = self._request("GET", f"repos/{self.repo}/git/commits/{commit_sha}")
        if status == 200:
            return data
        return None

    def create_blob(self, content: bytes) -> Optional[str]:
        """创建blob对象，返回blob SHA"""
        encoded = base64.b64encode(content).decode("utf-8")
        status, data = self._request(
            "POST",
            f"repos/{self.repo}/git/blobs",
            {"content": encoded, "encoding": "base64"},
        )
        if status == 201:
            return data.get("sha")
        print(f"    ⚠️ Blob创建失败: status={status}, error={data}")
        return None

    def create_tree(self, base_tree_sha: str, tree_items: List[dict]) -> Optional[dict]:
        """创建tree对象"""
        status, data = self._request(
            "POST",
            f"repos/{self.repo}/git/trees",
            {"base_tree": base_tree_sha, "tree": tree_items},
        )
        if status == 201:
            return data
        return None

    def create_commit(
        self,
        message: str,
        tree_sha: str,
        parent_sha: str,
    ) -> Optional[dict]:
        """创建commit对象"""
        status, data = self._request(
            "POST",
            f"repos/{self.repo}/git/commits",
            {
                "message": message,
                "tree": tree_sha,
                "parents": [parent_sha],
            },
        )
        if status == 201:
            return data
        return None

    def update_ref(self, branch: str, commit_sha: str) -> Optional[dict]:
        """更新分支引用"""
        status, data = self._request(
            "PATCH",
            f"repos/{self.repo}/git/refs/heads/{branch}",
            {"sha": commit_sha, "force": False},
        )
        if status == 200:
            return data
        return None

    def create_initial_commit(
        self,
        message: str,
        tree_sha: str,
    ) -> Optional[dict]:
        """创建初始commit（无parent）"""
        status, data = self._request(
            "POST",
            f"repos/{self.repo}/git/commits",
            {
                "message": message,
                "tree": tree_sha,
                "parents": [],
            },
        )
        if status == 201:
            return data
        return None


def collect_files(project_root: str, gitignore_path: Optional[str] = None) -> List[Tuple[str, bytes]]:
    """收集需要提交的文件"""
    ignore_patterns = [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".DS_Store",
        ".idea",
        ".vscode",
        ".dart_tool",
        ".flutter-plugins",
        "build/",
        "dist/",
        "node_modules/",
        "*.log",
        "outputs/",
    ]

    if gitignore_path and os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        ignore_patterns.append(line)
        except:
            pass

    files = []
    project_root = os.path.abspath(project_root)

    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in [".git"]]

        for filename in filenames:
            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, project_root).replace("\\", "/")

            should_ignore = False
            for pat in ignore_patterns:
                if pat.endswith("/"):
                    if pat.rstrip("/") in relpath.split("/"):
                        should_ignore = True
                        break
                elif pat.startswith("*"):
                    if relpath.endswith(pat[1:]):
                        should_ignore = True
                        break
                elif pat in relpath.split("/"):
                    should_ignore = True
                    break

            if should_ignore:
                continue

            try:
                with open(filepath, "rb") as f:
                    content = f.read()

                try:
                    content.decode("utf-8")
                except:
                    if len(content) > 10 * 1024 * 1024:
                        print(f"  [跳过] 大文件 (>10MB): {relpath}")
                        continue

                files.append((relpath, content))
            except Exception as e:
                print(f"  [警告] 无法读取文件 {relpath}: {e}")

    return files


def push_to_github(
    token: str,
    repo: str,
    message: str,
    project_root: str,
) -> bool:
    """推送代码到GitHub仓库"""
    api = GitHubAPI(token, repo)

    print("=" * 60)
    print(f"🚀 开始提交代码到 GitHub: {repo}")
    print("=" * 60)

    print("\n[1/5] 检查仓库...")
    if not api.check_repo_exists():
        print(f"  ❌ 仓库不存在或无权限访问: {repo}")
        print("  请确认:")
        print("    - Token 包含 repo 权限")
        print("    - 仓库地址正确")
        return False
    print("  ✅ 仓库存在，权限正常")

    branch = api.get_default_branch() or "main"
    print(f"  📌 默认分支: {branch}")

    print(f"\n[2/5] 收集项目文件...")
    files = collect_files(project_root, os.path.join(project_root, ".gitignore"))
    print(f"  📄 共收集 {len(files)} 个文件")
    for relpath, _ in files:
        size = len(_)
        size_str = f"{size/1024:.1f}KB" if size >= 1024 else f"{size}B"
        print(f"     - {relpath} ({size_str})")

    print(f"\n[3/5] 创建文件 Blob...")
    ref_info = api.get_latest_commit(branch)
    base_tree_sha = None
    parent_sha = None
    is_initial = False

    if ref_info and ref_info.get("object", {}).get("sha"):
        parent_sha = ref_info["object"]["sha"]
        commit_info = api.get_commit(parent_sha)
        if commit_info:
            base_tree_sha = commit_info.get("tree", {}).get("sha")
        print(f"  📍 已有最新commit: {parent_sha[:8]}...")
    else:
        print("  🆕 仓库为空，将创建初始commit")
        is_initial = True

    tree_items = []
    success_count = 0
    for i, (relpath, content) in enumerate(files):
        blob_sha = api.create_blob(content)
        if blob_sha:
            tree_items.append(
                {
                    "path": relpath,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha,
                }
            )
            success_count += 1
            if (i + 1) % 5 == 0 or i == len(files) - 1:
                print(f"  ⏳ 进度: {success_count}/{len(files)}")
        else:
            print(f"  ❌ 创建blob失败: {relpath}")

    if success_count == 0:
        print("  ❌ 没有文件成功创建blob")
        return False

    print(f"  ✅ {success_count} 个文件blob创建成功")

    print(f"\n[4/5] 创建 Commit...")
    tree_result = api.create_tree(base_tree_sha or "", tree_items)
    if not tree_result:
        print("  ❌ 创建tree失败")
        return False
    tree_sha = tree_result["sha"]
    print(f"  🌳 Tree SHA: {tree_sha[:12]}...")

    if is_initial:
        commit_result = api.create_initial_commit(message, tree_sha)
    else:
        commit_result = api.create_commit(message, tree_sha, parent_sha)

    if not commit_result:
        print("  ❌ 创建commit失败")
        return False
    commit_sha = commit_result["sha"]
    print(f"  📝 Commit SHA: {commit_sha[:12]}...")
    print(f"  💬 提交信息: {message}")

    print(f"\n[5/5] 更新分支引用...")
    if is_initial:
        status, result = api._request(
            "POST",
            f"repos/{repo}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": commit_sha},
        )
        update_ok = status == 201
    else:
        update_result = api.update_ref(branch, commit_sha)
        update_ok = update_result is not None

    if not update_ok:
        print("  ❌ 更新分支引用失败")
        return False
    print(f"  ✅ 分支 {branch} 已更新")

    print("\n" + "=" * 60)
    print("🎉 提交成功!")
    print("=" * 60)
    print(f"  仓库地址: https://github.com/{repo}")
    print(f"  Commit: {commit_sha}")
    print(f"  文件数: {success_count}")
    print(f"  分支: {branch}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="通过GitHub API提交代码")
    parser.add_argument("--token", required=False, help="GitHub Token (需要repo权限)")
    parser.add_argument("--repo", default="Idol993/rh001", help="仓库地址 (owner/repo)")
    parser.add_argument("--message", "-m", default=None, help="提交信息")
    parser.add_argument("--round", "-r", type=int, default=1, help="轮次编号")
    parser.add_argument("--path", default=None, help="项目根目录")

    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("❌ 请提供 GitHub Token:")
        print("   方式1: --token ghp_xxx")
        print("   方式2: 环境变量 GITHUB_TOKEN")
        print("\n获取Token: https://github.com/settings/tokens")
        sys.exit(1)

    message = args.message or f"round-{args.round}: 提交代码 - 智能炒股分析APP核心逻辑"

    project_root = args.path
    if not project_root:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    success = push_to_github(token, args.repo, message, project_root)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
