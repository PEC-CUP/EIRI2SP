#!/usr/bin/env python
"""Parse a GC-EI-MS mzML file into RT, TIC, and per-scan spectra."""

from __future__ import annotations

import argparse
import base64
import struct
import zlib
from pathlib import Path

import numpy as np
from lxml import etree


NS = {"m": "http://psi.hupo.org/ms/mzml"}


def decode_binary(elem: etree._Element) -> np.ndarray:
    """Decode a mzML binaryDataArray into a numpy array."""
    compressed = elem.find('.//m:cvParam[@accession="MS:1000574"]', NS) is not None
    is64 = elem.find('.//m:cvParam[@accession="MS:1000523"]', NS) is not None
    text = elem.findtext("m:binary", namespaces=NS) or ""
    raw = base64.b64decode(text.strip())
    if compressed:
        raw = zlib.decompress(raw)
    dtype = "<f8" if is64 else "<f4"
    return np.frombuffer(raw, dtype=dtype)


def parse(path: str) -> tuple[np.ndarray, list[dict]]:
    rts: list[float] = []
    tics: list[float] = []
    spectra: list[dict] = []
    for event, elem in etree.iterparse(path, events=("end",), tag="{http://psi.hupo.org/ms/mzml}spectrum"):
        rt = None
        mz = None
        intensity = None
        for ba in elem.findall("m:binaryDataArrayList/m:binaryDataArray", NS):
            kind = None
            for c in ba.findall("m:cvParam", NS):
                acc = c.get("accession")
                if acc == "MS:1000514":
                    kind = "mz"
                elif acc == "MS:1000515":
                    kind = "intensity"
            arr = decode_binary(ba)
            if kind == "mz":
                mz = arr
            elif kind == "intensity":
                intensity = arr
        for c in elem.iter("{http://psi.hupo.org/ms/mzml}cvParam"):
            if c.get("accession") == "MS:1000016":
                rt = float(c.get("value"))
                break
        if rt is None or mz is None or intensity is None:
            elem.clear()
            continue
        tic = float(intensity.sum()) if intensity.size else 0.0
        rts.append(rt)
        tics.append(tic)
        spectra.append({"rt": rt, "mz": mz, "intensity": intensity})
        elem.clear()
    return np.array(rts), np.array(tics), spectra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE / "data/exp/33_sp100.mzML"))
    parser.add_argument("--out-dir", default=str(BASE / "data/exp"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rts, tics, spectra = parse(args.input)
    np.save(out / "rts.npy", rts)
    np.save(out / "tics.npy", tics)
    # 保存每个谱图的 m/z 和 intensity（等长补齐到 m/z 0-500 整数 bin）
    bins = np.arange(0, 501, dtype=np.float64)
    ei_matrix = np.zeros((len(spectra), len(bins)), dtype=np.float32)
    for i, s in enumerate(spectra):
        idx = np.round(s["mz"]).astype(int)
        mask = (idx >= 0) & (idx <= 500)
        np.add.at(ei_matrix[i], idx[mask], s["intensity"][mask].astype(np.float32))
    # 归一化到最大峰
    mx = ei_matrix.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    ei_matrix /= mx
    np.save(out / "ei_matrix.npy", ei_matrix)
    print(f"scans={len(rts)}, RT {rts.min():.2f}-{rts.max():.2f} min, TIC sum={tics.sum():.3e}")
    print(f"EI matrix shape={ei_matrix.shape}")


if __name__ == "__main__":
    main()
