#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Windows 的 DPAPI 保护本机密钥 —— 一个不认识本项目的纯平台模块。

为什么需要它
------------
`ai_config.json` 里原本是**明文密钥**，而它已经真的漏出去过一次：那个文件一直被 git
跟踪，`dist/` 里两个手工打的 zip 又各带了一份，谁下载这个软件谁就拿到一把能直接打
`api.deepseek.com` 花掉余额的钥匙（见 CHANGELOG）。当时补的三道防线 —— `.gitignore`
挡住、`git rm --cached` 摘掉、`打包.py` 打完复检 —— 防的都是「**这一份**别出去」，
防不住「它已经出去了」：目录被拷走、进了备份、换一种打包方式、截图，穿过任何一样，
拿到的都是一把能直接用的钥匙。

DPAPI（`CryptProtectData`）把密文**绑到当前 Windows 账户**上：同一个文件拷到别的机器、
别的账户底下就是一串解不开的字节。系统自带的接口，`ctypes` 直接调，**不需要任何依赖**
—— 本项目「不引 CDN、不引框架、离线可用」那条底线在这里同样成立。

它挡什么、不挡什么（这句要紧，别读成「密钥从此安全了」）
------------------------------------------------------
- **挡**：文件被拷到别处 —— 换台机器、换个 Windows 账户、进了备份、被打进包里发出去。
- **不挡**：已经以你的身份在运行的程序。那种程序自己也能调 `CryptUnprotectData`，
  DPAPI 对它一点阻力都没有。本机任意进程都能访问 `127.0.0.1` 这件事没有改变
  （见 `server.py` 头部：不再往上加权限）。

所以这个模块的定位是「把已经发生过的那次泄漏变成无害」，**不是**「密钥从此安全」。

失败一律返回 None，绝不抛异常
-----------------------------
`available()` / `protect()` / `unprotect()` 都不抛。叫不动 DPAPI 的机器（非 Windows、
crypt32 出问题、密文被改坏）是降级路径的**正常输入**，由调用方据此退回明文存储并
**明确告知用户**（`ai.py` 的 `save_config` / `status`）。一次加密失败不该让「保存配置」
这个动作整个失败 —— 那只会逼用户去手写明文文件。
"""

from __future__ import annotations

import base64
import ctypes
import sys

# 固定附加熵。别的程序要解这份密文，光调 CryptUnprotectData 不够，还得知道这串字节。
# 它挡不住有心人（就写在源码里），但能让「随手写个脚本试试」落空。
# **改了它，已经存下的密文就再也解不开** —— 当成格式版本号看待。
APP_ENTROPY = b"fjc-chain/ai_config"

# 绝不弹系统对话框。服务是后台跑的，弹窗没人点，只会把进程卡死在那里。
CRYPTPROTECT_UI_FORBIDDEN = 0x1

_IS_WINDOWS = sys.platform == "win32"

# 用 c_uint32 / c_char 拼 DATA_BLOB，**不 import ctypes.wintypes** ——
# 那个模块在非 Windows 上会因为 WINFUNCTYPE 不存在而直接 ImportError，
# 而这个文件必须能在任何平台上被 import（测试、非 Windows 部署都要走降级路径）。
class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32),
                ("pbData", ctypes.POINTER(ctypes.c_char))]


_LIB = None
_LIB_TRIED = False


def _crypt32():
    """拿到配好原型的 crypt32；拿不到返回 None。只在 Windows 上会成功。"""
    global _LIB, _LIB_TRIED
    if _LIB_TRIED:
        return _LIB
    _LIB_TRIED = True
    if not _IS_WINDOWS:
        return None
    try:
        lib = ctypes.WinDLL("crypt32", use_last_error=True)

        lib.CryptProtectData.argtypes = [
            ctypes.POINTER(_Blob), ctypes.c_wchar_p, ctypes.POINTER(_Blob),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(_Blob)]
        lib.CryptProtectData.restype = ctypes.c_int

        # 第二个参数（描述串）传 NULL —— 它是可选的，而填了就得自己 LocalFree，
        # 白白多一条泄漏路径。我们不需要那个描述。
        lib.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob),
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(_Blob)]
        lib.CryptUnprotectData.restype = ctypes.c_int

        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB


def _local_free(ptr) -> None:
    """CryptProtectData/CryptUnprotectData 用 LocalAlloc 分配，得还回去。"""
    try:
        ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(
            ctypes.cast(ptr, ctypes.c_void_p))
    except Exception:
        pass


def _make_blob(data: bytes) -> tuple[_Blob, object]:
    """返回 (blob, 持有底层内存的对象)。

    第二个返回值必须被调用方**一直引用着** —— `_Blob` 里只有一个裸指针，
    ctypes 不会替我们保住那块 buffer 的命。
    """
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf


def _blob_bytes(blob: _Blob) -> bytes:
    if not blob.pbData or not blob.cbData:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)


def available() -> bool:
    """这台机器上能不能用 DPAPI。结果缓存 —— 平台不会中途变。"""
    return _crypt32() is not None


def protect(plain: str) -> str | None:
    """明文 → base64 密文。失败返回 None（调用方据此退回明文存储）。"""
    lib = _crypt32()
    if lib is None or not plain:
        return None
    try:
        blob_in, _keep_in = _make_blob(plain.encode("utf-8"))
        ent, _keep_ent = _make_blob(APP_ENTROPY)
        out = _Blob()
        ok = lib.CryptProtectData(
            ctypes.byref(blob_in), "fjc-chain ai_config key", ctypes.byref(ent),
            None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
        if not ok:
            return None
        try:
            return base64.b64encode(_blob_bytes(out)).decode("ascii")
        finally:
            _local_free(out.pbData)
    except Exception:
        return None


def unprotect(b64: str) -> str | None:
    """base64 密文 → 明文。失败返回 None —— 换过 Windows 账户或换过机器就会走这里。"""
    lib = _crypt32()
    if lib is None or not b64:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return None
    if not raw:
        return None
    try:
        blob_in, _keep_in = _make_blob(raw)
        ent, _keep_ent = _make_blob(APP_ENTROPY)
        out = _Blob()
        ok = lib.CryptUnprotectData(ctypes.byref(blob_in), None, ctypes.byref(ent),
                                    None, None, CRYPTPROTECT_UI_FORBIDDEN,
                                    ctypes.byref(out))
        if not ok:
            return None
        try:
            return _blob_bytes(out).decode("utf-8")
        finally:
            _local_free(out.pbData)
    except Exception:
        return None


if __name__ == "__main__":
    # 手动冒烟：python ai_secret.py
    print("平台:", sys.platform, "| Windows:", _IS_WINDOWS)
    print("available():", available())
    sample = "sk-self-test-0123456789"
    blob = protect(sample)
    if blob is None:
        print("protect() 返回 None —— 这台机器走降级路径（明文存储）")
    else:
        print("密文长度:", len(blob), "| 密文里有明文吗:", sample in blob)
        back = unprotect(blob)
        print("解回来对得上:", back == sample)
        print("改坏一个字符还解得开吗:", unprotect(blob[:-4] + "AAAA") is not None)
