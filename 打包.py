#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把项目打成一个可以发给别人的 zip —— 附一份**逐条检查过的**文件清单。

为什么有这个脚本
----------------
dist/ 里的 1.1.0.zip 和 1.1.1.zip 是手工打出来的，两个都**各带了一份
ai_config.json**，也就是把使用者的 DeepSeek 密钥连同软件一起发了出去。
手工打包没有清单、没有检查，漏一个文件不会有任何提示 —— 而这里漏掉的
恰好是最不该漏的那一个。

所以这个脚本把那一步固定成三件事：

  1. 一份**写在明处的排除清单**（密钥 / 字节码 / 锁文件 / dist 自己）；
  2. 打完**把 zip 重新打开逐条查一遍** —— 不只是「按清单排除了」，
     而是真的回去看产物里有什么；还会用本机凭据（ai_config.json 的 key、
     token.txt 的 token）的原文去扫包里每一个文本文件，确认它没被抄进
     README、测试或别处。查出问题就**删掉 zip 并以非零码退出**，
     不给「先发了再说」留缝；
  3. 打印最终清单与体积，发出去之前一眼能看完。

用法
----
    python 打包.py                    # 出 dist/2026-09-22.zip
    python 打包.py --version 1.2.0    # 出 dist/1.2.0.zip
    python 打包.py --list             # 只列清单，不写文件
    python 打包.py --with-slides      # 把汇报 pptx 也打进去（默认不打）
    python 打包.py --force            # 覆盖同名 zip（默认拒绝）

版本号没有单一来源（代码里没有 __version__，历史上只体现在 zip 文件名上），
所以这里**不猜**：默认用当天日期，要发版本号就显式传 --version。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))

try:
    import ai_secret      # 只为把 DPAPI 密文解回原文用（见 local_secrets）
except Exception:         # 拿不到就是少扫一条，不该让打包整个失败
    ai_secret = None

# 排除清单分三类写，改的时候知道自己动的是哪一类。
SECRET_FILES = {"ai_config.json", "token.txt", ".gh_token"}   # 1) 凭据：这个脚本存在的理由
JUNK_DIRS = {"__pycache__", "dist", ".git", ".claude", ".idea", ".vscode"}
JUNK_SUFFIXES = (".pyc", ".pyo")                       # 2) 生成物 / 版本控制 / 编辑器残留
JUNK_PREFIXES = ("~$",)                                #    Office 锁文件
SLIDES_SUFFIXES = (".pptx", ".ppt", ".key")            # 3) 体积大、与运行无关（可放回）

# 二进制不需要按文本扫 key
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
                   ".zip", ".woff", ".woff2", ".ttf", ".pdf")


def wanted(rel: str, with_slides: bool) -> bool:
    """这个相对路径要不要进包。"""
    parts = rel.replace("\\", "/").split("/")
    if any(p in JUNK_DIRS for p in parts):
        return False
    base = parts[-1]
    if base in SECRET_FILES or base == ".gitignore":
        return False
    if base.startswith(JUNK_PREFIXES) or base.endswith(JUNK_SUFFIXES):
        return False
    if not with_slides and base.lower().endswith(SLIDES_SUFFIXES):
        return False
    return True


def collect(with_slides: bool) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in JUNK_DIRS]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
            if wanted(rel, with_slides):
                out.append(rel)
    return sorted(out)


def key_texts(cfg) -> list[str]:
    """从一个配置字典里抠出所有形态的密钥原文：明文直接有，密文要先解开。

    **加密之后 `.get("key")` 就永远是 None 了。** 照旧只读它，这道检查会
    **静默失效** —— 少扫一条不报错、不告警，只是安静地放行，而那正是它存在的理由
    （「凭据被抄进别处」只有这一条查得出来）。所以两个字段都得管：
    老文件是明文，新文件是 `key_dpapi`，两种都可能出现在这台机器上。
    """
    if not isinstance(cfg, dict):
        return []
    out = []
    plain = cfg.get("key")
    if isinstance(plain, str) and len(plain) >= 8:
        out.append(plain)
    blob = cfg.get("key_dpapi")
    if isinstance(blob, str) and blob.strip() and ai_secret is not None:
        try:
            back = ai_secret.unprotect(blob.strip())
        except Exception:
            back = None
        if isinstance(back, str) and len(back) >= 8:
            out.append(back)
    return out


def local_secrets() -> list[str]:
    """本机凭据的原文：ai_config.json 里的 key + token.txt 里的 token。

    读不出来就跳过 —— 少扫一条不代表放行，只是那一项没得比。
    两份都要扫，因为泄漏从来不看文件叫什么名字：真出事的是「那段文本
    出现在了包里」，而不是「那个文件在包里」。
    """
    out = []
    try:
        with open(os.path.join(ROOT, "ai_config.json"), encoding="utf-8") as fh:
            out.extend(key_texts(json.load(fh)))
    except Exception:
        pass
    try:
        with open(os.path.join(ROOT, "token.txt"), encoding="utf-8") as fh:
            v = fh.read().strip()
        if len(v) >= 8:
            out.append(v)
    except Exception:
        pass
    return out


def verify(zip_path: str, secrets: list[str]) -> list[str]:
    """把 zip 重新打开逐条查。返回问题列表，空列表 = 干净。"""
    problems = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            rel = info.filename
            base = rel.rsplit("/", 1)[-1]
            if base in SECRET_FILES:
                problems.append(f"{rel}：凭据文件不该进包")
            if "__pycache__" in rel or base.endswith(JUNK_SUFFIXES):
                problems.append(f"{rel}：字节码缓存")
            if base.startswith(JUNK_PREFIXES):
                problems.append(f"{rel}：Office 锁文件")
            # 最后这道是防「凭据被抄进别处」：名字查不出来，只有内容扫得出来。
            if secrets and info.file_size and not base.lower().endswith(BINARY_SUFFIXES):
                try:
                    text = z.read(rel).decode("utf-8", "ignore")
                except Exception:
                    continue
                for s in secrets:
                    if s in text:
                        problems.append(f"{rel}：**里面出现了本机凭据原文**")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="打包成可以发给别人的 zip（默认排除密钥、字节码、锁文件）")
    ap.add_argument("--version", help="版本号，决定文件名；默认用当天日期")
    ap.add_argument("--out", default="dist", help="输出目录，默认 dist/")
    ap.add_argument("--with-slides", action="store_true",
                    help="把 .pptx 汇报稿也放进去（默认不放：5MB，与运行无关）")
    ap.add_argument("--list", action="store_true", help="只打印清单，不写 zip")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的同名 zip")
    args = ap.parse_args(argv)

    files = collect(args.with_slides)
    if not files:
        print("没有可打包的文件，检查一下是不是在项目根目录。", file=sys.stderr)
        return 1

    total = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
    print(f"选中 {len(files)} 个文件，原始体积 {total / 1048576:.1f} MB")
    for f in files:
        print(f"  {f}")

    skipped = [f for f in collect(True) if f not in set(files)]
    if skipped:
        print(f"\n排除 {len(skipped)} 项：")
        for f in skipped:
            print(f"  {f}")

    if args.list:
        return 0

    version = args.version or _dt.date.today().isoformat()
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"{version}.zip")

    if os.path.exists(zip_path) and not args.force:
        print(f"\n{zip_path} 已存在。换个 --version，或者加 --force 覆盖。",
              file=sys.stderr)
        return 1

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(os.path.join(ROOT, rel), rel)

    secrets = local_secrets()
    problems = verify(zip_path, secrets)
    if problems:
        os.remove(zip_path)
        print("\n产物有问题，已删除，没有留下半成品：", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    size = os.path.getsize(zip_path)
    print(f"\n{zip_path}")
    print(f"  {len(files)} 个文件，{size / 1048576:.1f} MB"
          f"（压缩率 {100 * (1 - size / total):.0f}%）")
    print("  复检通过：无凭据文件、无字节码、无锁文件"
          + (f"；已用本机 {len(secrets)} 条凭据原文扫过全部文本内容" if secrets else ""))
    if not args.with_slides:
        print("  注：汇报 pptx 没打进去，要的话加 --with-slides。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
