"""Read-only Agent configuration diagnostics; never prints configuration values.

Run: PlanAgent/.venv/bin/python scripts/check_agent_env.py
No network requests are made; successful checks do not verify key validity.
"""

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "AMAP_KEY", "TAVILY_API_KEY")


def inspect_file(path):
    print(f"\n文件：{path.relative_to(ROOT)}")
    if not path.exists():
        print("  [提示] 文件不存在（仍可能从进程环境或回退文件取得配置）")
        return {}
    try:
        raw = path.read_text(encoding="utf-8-sig")
        values = dict(dotenv_values(stream=__import__("io").StringIO(raw)))
    except (OSError, UnicodeError):
        print("  [错误] 无法读取文件或文件不是 UTF-8 编码")
        return {}
    seen = set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if not match:
            print(f"  [错误] 第 {number} 行不是 KEY=value 格式")
            continue
        key = match[1]
        if key in seen:
            print(f"  [警告] 第 {number} 行重复定义 {key}，后面的定义覆盖前面的定义")
        seen.add(key)
        if any(re.search(r"\b" + field + r"\s*=", line[match.end():]) for field in FIELDS):
            print(f"  [错误] 第 {number} 行疑似拼接了多个配置项，请每行只写一项")
    print("  已读取；所有配置值均隐藏")
    return values


def main():
    print("只读检查，不发送请求，不验证密钥有效性。进程环境变量优先于 .env。")
    files = {name: inspect_file(ROOT / name / ".env") for name in
             ("QuestionAgent", "PlanAgent", "SearchAgent", "ValidateAgent")}
    errors = 0
    for agent in ("QuestionAgent", "PlanAgent", "SearchAgent", "ValidateAgent"):
        print(f"\n[{agent} 有效配置]")
        sources = [("进程环境变量", os.environ), (f"{agent}/.env", files[agent])]
        if agent == "QuestionAgent":
            sources.append(("PlanAgent/.env（QuestionAgent 回退配置）", files["PlanAgent"]))
        fields = list(FIELDS[:3])
        if agent in ("PlanAgent", "SearchAgent"):
            fields.append("AMAP_KEY")
        if agent == "SearchAgent":
            fields.append("TAVILY_API_KEY")
        for field in fields:
            source, value = next(((label, data[field]) for label, data in sources
                                  if field in data and data[field] is not None), (None, None))
            if not value or not value.strip():
                required = field == "OPENAI_API_KEY"
                print(f"  [{'错误' if required else '提示'}] {field} 未设置或为空"
                      + (f"；来源：{source}" if source else "")
                      + ("；将使用代码/SDK默认值" if field in FIELDS[1:3] else ""))
                errors += int(required)
                continue
            if field == "OPENAI_BASE_URL":
                url = urlsplit(value)
                if url.scheme not in ("http", "https") or not url.netloc:
                    print(f"  [错误] {field} 地址格式无效；来源：{source}")
                    errors += 1
                    continue
            if any(word in value for word in ("你的", "your-key", "YOUR_API_KEY", "<API")):
                print(f"  [错误] {field} 仍是占位内容；来源：{source}")
                errors += 1
            else:
                print(f"  [已设置] {field}；来源：{source}（值隐藏）")
    print(f"\n有效配置错误数：{errors}。文件格式问题请同时查看上方逐行报告。")
    print("注意：本脚本检查当前终端环境；已启动后端继承的环境变量可能不同。")
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
