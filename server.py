"""自由连接链（FJC）计算器 —— Flask 后端。

页面：GET  /
接口：POST /api/compute   {"n": 100 | [10,100,1000], "l": 1.0}
      POST /api/chain     {"n": 1000, "l": 1.0, "seed": 12345, "chains": 1}
      POST /api/sweep     {"n": [10,30,100,300,1000,3000], "chains": 10000}
      POST /api/force     {"n": 100, "l": 1.0, "x_max": 10, "temperature": 298.15}
      POST /api/solve_n   {"h": 37.0, "l": 1.0, "kind": "h_rms"}
      GET  /api/kuhn      # Kuhn 长度预置表（唯一的那份在 fjc_core.KUHN_PRESETS）

      AI 助手：
      GET  /api/ai/status
      POST /api/ai/config    {"key"?, "endpoint"?, "model"?, "clear_key"?}
      POST /api/ai/test      {"key"?, "endpoint"?, "model"?}
      POST /api/ai/ask       {"message": "...", "session_id"?: "..."}
      POST /api/ai/resume    {"session_id": "...", "results": [...]}

物理与链模拟全部在 fjc_core.py 里，AI 那条链全部在 ai.py 里。
本文件只负责 HTTP、端口，以及把请求体检查到能给下面用的程度。
"""

from __future__ import annotations

import argparse
import socket
import sys
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import ai
from fjc_core import (
    CHAIN_POINTS_MAX,
    CURVE_POINTS,
    DEFAULT_TEMPERATURE,
    FJCInputError,
    MAX_CHAINS,
    SWEEP_DEFAULT_CHAINS,
    compute_many,
    force_extension,
    kuhn_presets,
    random_chain,
    solve_n,
    sweep,
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DEFAULT_PORT = 8770
MAX_CURVES = 5

# 一条提问的字数上限。真实提问都在一两百字，给到两千已经很宽松了；
# 再长多半是误粘贴，而且它会被原样转发给按 token 计费的网关。
AI_MAX_MESSAGE_CHARS = 2000

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.post("/api/compute")
def api_compute():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="请求体必须是 JSON 对象"), 400

    raw_n = payload.get("n")
    # 允许 n 是单个数，也允许是数组（用于多条曲线对比）
    ns = raw_n if isinstance(raw_n, list) else [raw_n]
    if len(ns) > MAX_CURVES:
        return jsonify(error=f"最多同时对比 {MAX_CURVES} 条曲线，收到 {len(ns)} 个 n"), 400

    try:
        results = compute_many(ns, payload.get("l", 1.0), CURVE_POINTS)
    except FJCInputError as e:
        return jsonify(error=str(e)), 400

    return jsonify(results=[r.to_dict() for r in results])


@app.post("/api/chain")
def api_chain():
    """单链构象模拟：随机生成 chains 条 FJC 链的顶点，供 3D 视图绘制。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="请求体必须是 JSON 对象"), 400

    try:
        result = random_chain(
            payload.get("n"),
            payload.get("l", 1.0),
            payload.get("seed"),
            payload.get("chains", 1),
        )
    except FJCInputError as e:
        return jsonify(error=str(e)), 400

    return jsonify(result.to_dict())


@app.post("/api/sweep")
def api_sweep():
    """链长扫描：一组 n，每个采 chains 条独立链，给出 ⟨R²⟩ 对 n 的标度关系。

    比 /api/chain 慢得多（默认 M=10000 时约 3 秒）—— 因为它算的是统计量，
    不是一条链的几何。前端要在这期间显示忙状态。
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="请求体必须是 JSON 对象"), 400

    try:
        result = sweep(
            payload.get("n"),
            payload.get("l", 1.0),
            payload.get("seed"),
            payload.get("chains", SWEEP_DEFAULT_CHAINS),
        )
    except FJCInputError as e:
        return jsonify(error=str(e)), 400

    return jsonify(result.to_dict())


@app.post("/api/force")
def api_force():
    """力–伸长曲线：λ(x) = Langevin(x)，横轴是无量纲力 f·l/(k_BT)。

    n、l 只影响返回里的绝对伸长那一列，**λ 曲线本身与二者无关** ——
    这是这条曲线的定义性性质，也是它值得单画一张图的理由。
    temperature 只影响换算成 pN 的那一列。
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="请求体必须是 JSON 对象"), 400

    try:
        result = force_extension(
            payload.get("n"),
            payload.get("l", 1.0),
            payload.get("x_max", 10.0),
            CURVE_POINTS,
            DEFAULT_TEMPERATURE if payload.get("temperature") is None
            else payload["temperature"],
        )
    except FJCInputError as e:
        return jsonify(error=str(e)), 400

    return jsonify(result.to_dict())


@app.post("/api/solve_n")
def api_solve_n():
    """由某个特征末端距反解链段数。

    `kind` 必须说清「哪一种 h」：h_rms / ⟨h⟩ / h* / nl 在同一个 n 下能差
    一到两个数量级，问「要 100」而不说哪种 100，答案没意义。默认 h_rms。
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="请求体必须是 JSON 对象"), 400

    try:
        result = solve_n(payload.get("h"), payload.get("l", 1.0),
                         payload.get("kind", "h_rms"))
    except FJCInputError as e:
        return jsonify(error=str(e)), 400

    return jsonify(result.to_dict())


@app.get("/api/kuhn")
def api_kuhn():
    """Kuhn 长度预置表。

    **唯一的那份在 `fjc_core.KUHN_PRESETS`** —— 前端照这个生成下拉，
    不在 JS 里再抄一份（否则改了一处忘另一处，两个表就各说各话了）。
    每条都带可信度标签，因为这些数字是**量级参考、不是权威值**。
    """
    return jsonify(presets=kuhn_presets(), note=(
        "b = 2p 是定义，没有余地；各聚合物的 b 取值在文献里同一种能差一倍"
        "（温度、立构规整度、溶剂、以及作者给的到底是 p 还是 b）。"
        "这张表用来把 l 填到正确的量级，不是替代文献。"
    ))


@app.errorhandler(404)
def not_found(_e):
    return jsonify(error="没有这个地址"), 404


# --- AI 助手 -------------------------------------------------------------
# 这几个路由都不写业务逻辑，只做「请求体够不够用」的检查，剩下全交给 ai.py。
#
# 密钥相关的四点，改这里之前先读一遍：
#   1. **写配置的路由（/api/ai/config）是后来才加的。** 最初的决定是「密钥不进浏览器」，
#      理由是 fjc-chain 本地单用户、零认证，任何能访问 127.0.0.1:8770 的进程都能打到
#      这些接口上，再加一个写 key 的入口等于把密钥的写入权限交给本机任意程序。
#      后来按用户要求加上了 —— 手改 JSON 对日常换 key 太笨。**代价是认下来的，不是忘了**：
#      别再往上加权限（认证、多用户、远程访问），加了这条理由就会变成真问题。
#   2. /api/ai/status 和 /api/ai/config 的返回值都只带掩码，**原始 key 永不出现在任何
#      响应里** —— 包括测试连接失败时网关回显 key 的那种错误（ai.scrub 会洗）。
#   2b. **密钥在磁盘上是加密的**（Windows DPAPI，见 ai_secret.py）—— 落盘那一步在
#      ai.save_config 里，路由这层看不到明文也看不到密文。存的只是一份「能不能加密」
#      的标记（`status.key_storage`），因为降级到明文**必须让用户知道**。
#      `main()` 启动时调一次 ai.migrate_config()，把老文件里的明文顺手换成密文。
#      DPAPI 挡的是「文件被拷到别处」，**挡不住**已经以你的身份在跑的程序 ——
#      所以上面第 1 条的理由（本机任意进程都能打这些接口）一点没变，别读成「现在安全了」。
#   3. 助手（模型）**不能**调这两个路由：能改 endpoint 就等于能把你下一问连同密钥
#      一起发到它挑的地方。它们只由用户点按钮触发，也不在 ai_tools 的工具表里。
#
# CSRF 面靠 `request.get_json(silent=True)`：它只认 application/json，
# 跨站表单那种 form-encoded 的 POST 会拿到 None，直接 400；
# 跨站 fetch 带 application/json 会先预检，而我们不发任何 CORS 头，浏览器直接拦。

def _ai_payload() -> tuple[dict | None, tuple | None]:
    """取并粗检请求体。返回 (payload, 错误响应)。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, (jsonify(error="请求体必须是 JSON 对象"), 400)
    return payload, None


def _ai_session_id(payload: dict) -> str | None:
    sid = payload.get("session_id")
    if sid is None:
        return None
    if not isinstance(sid, str) or len(sid) > 64:
        return ""
    return sid


@app.get("/api/ai/status")
def api_ai_status():
    """助手配置的只读视图。前端用它决定显示「已配置」还是「本地模式」，
    也用它里面的 providers 填「选择服务商」下拉。"""
    return jsonify(ai.status())


@app.post("/api/ai/config")
def api_ai_config():
    """保存页面上填的接入配置，写回 ai_config.json。

    字段语义在 `ai.save_config` 的 docstring 里（key 空 = 不改，要清得显式传
    clear_key）。这里只管请求体和异常的翻译 —— 校验、原子写、环境变量告警都在那边。
    """
    payload, err = _ai_payload()
    if err:
        return err

    try:
        out = ai.save_config(
            key=payload.get("key"),
            endpoint=payload.get("endpoint"),
            model=payload.get("model"),
            clear_key=bool(payload.get("clear_key")),
        )
    except ai.ConfigError as e:
        return jsonify(error=str(e)), 400
    return jsonify(out)


@app.post("/api/ai/test")
def api_ai_test():
    """对网关发一次最小请求，验密钥和地址通不通。**不带 tools**（理由见 ai.py）。

    没传的字段用当前已保存的值 —— 所以页面可以「改了还没保存」就先测。
    """
    payload, err = _ai_payload()
    if err:
        return err

    try:
        out = ai.test_connection(
            key=payload.get("key"),
            endpoint=payload.get("endpoint"),
            model=payload.get("model"),
        )
    except ai.ConfigError as e:
        return jsonify(error=str(e)), 400
    return jsonify(out)


@app.post("/api/ai/ask")
def api_ai_ask():
    payload, err = _ai_payload()
    if err:
        return err

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="message 必须是非空字符串"), 400
    if len(message) > AI_MAX_MESSAGE_CHARS:
        return jsonify(error=f"问题太长了（上限 {AI_MAX_MESSAGE_CHARS} 字）"), 400

    sid = _ai_session_id(payload)
    if sid == "":
        return jsonify(error="session_id 必须是字符串"), 400

    out = ai.ask(message, sid)
    return _ai_response(out)


@app.post("/api/ai/resume")
def api_ai_resume():
    """浏览器执行完前端工具后回调这里，把那一轮跑完。"""
    payload, err = _ai_payload()
    if err:
        return err

    sid = _ai_session_id(payload)
    if not sid:
        return jsonify(error="resume 必须带 session_id"), 400

    results = payload.get("results")
    if results is not None and not isinstance(results, list):
        return jsonify(error="results 必须是数组"), 400
    if isinstance(results, list) and len(results) > 16:
        return jsonify(error="results 一次最多 16 条"), 400

    out = ai.resume(sid, results or [])
    return _ai_response(out)


def _ai_response(out: dict) -> tuple:
    """助手响应的统一出口。

    **一律 200**（除了上面那些请求体本身就不对的 400），哪怕 ok 是 false：
    网关挂掉时本地兜底给出的那段话就在 body 里，那是要显示给用户看的答案；
    回 5xx 会让中间层（代理、fetch 封装、日志采样）把它当成「没有内容」丢掉。
    ok / source / degraded 三个字段已经把事情说清楚了。
    唯一例外是一条答案都给不出来（source == "none"）—— 那才是真的失败。
    """
    code = 502 if out.get("source") == "none" else 200
    return jsonify(out), code


def pick_port(start: int = DEFAULT_PORT, tries: int = 50) -> int:
    """从 start 起找一个能绑上的端口，避免写死端口撞车。"""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"从 {start} 起连续 {tries} 个端口都被占用，起不来了")


def main() -> None:
    parser = argparse.ArgumentParser(description="自由连接链（FJC）计算器")
    parser.add_argument("--port", type=int, default=None,
                        help=f"指定端口（默认从 {DEFAULT_PORT} 起自动找空位）")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    args = parser.parse_args()

    # 控制台编码兜底：万一有字符打不出来也不要崩
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    port = args.port or pick_port()
    url = f"http://127.0.0.1:{port}"
    print("")
    print("  自由连接链（FJC）计算器")
    print(f"  已启动：{url}")
    print("  按 Ctrl+C 停止")
    print("")

    # 把老 ai_config.json 里的明文密钥顺手换成密文（见 ai.migrate_config）。
    # **绝不能挡住启动** —— 配置有问题时用户要的正是「起得来，然后去页面上改」。
    # 没迁移（没什么可做 / 加不了密）时返回 None，安静跳过。
    try:
        note = ai.migrate_config()
    except Exception:
        note = None
    if note:
        print(f"  {note}")
        print("")

    if args.open:
        webbrowser.open(url)

    app.run(host="127.0.0.1", port=port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
