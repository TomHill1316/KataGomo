#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the four CPU-only B13/B16 packed FA4 artifacts for one geometry and a schema-2 bundle.

One bundle per shape-specialized profile. The attention geometry is supplied on
the command line so the same driver serves tf3-b11c768 (heads 12, packed row
1152) and b15 ffn-pruned (heads 16, packed row 1536).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from generate_fa4 import file_identity, verify_static_abi

HERE = Path(__file__).resolve().parent


def write_bundle(root, label="b11", heads=12, head_dim=32, sequence=361):
    """Assemble only after all four freshly generated objects pass structural checks."""
    root = Path(root).resolve()
    if (root / "bundle.json").exists():
        raise ValueError("refuse to overwrite an existing bundle manifest")
    qkv_stride = 3 * heads * head_dim
    output_stride = heads * head_dim
    entries = []
    for batch in (13,16):
        for mode in ("exact","mask"):
            prefix = f"{label}_fa4_static_b{batch}_{mode}_m128_n96_s1"
            directory = f"b{batch}/{mode}"
            path = root / directory / (prefix + ".json")
            item = json.loads(path.read_text())
            required = {"schema":2,"fixed_batch":batch,"symbol_prefix":prefix,"mode":mode,
                "tensor_abi":"pointer_only","qkv_layout":"packed","qkv_stride":qkv_stride,
                "output_stride":output_stride,"target":"sm120","io":"fp16",
                "runtime_batch":{"minimum":batch,"maximum":batch,"probe":batch},
                "fixed_shape":{"sequence":sequence,"heads":heads,"head_dim":head_dim},
                "tile":{"m":128,"n":96,"stages":1,"threads":128},
                "accumulation":{"qk":"fp16","pv":"fp16"}}
            if any(item.get(k) != v for k,v in required.items()):
                raise ValueError("unexpected static artifact profile: " + str(path))
            for suffix in (".o",".h","_api.h","_bridge.cpp"):
                actual = file_identity(path.parent / (prefix + suffix))["sha256"]
                if actual != item["artifacts"][suffix]["sha256"]:
                    raise ValueError("artifact hash mismatch: " + prefix + suffix)
            verify_static_abi((path.parent / (prefix + ".h")).read_text(),prefix,mode=="mask")
            entries.append({"batch":batch,"mode":mode,"directory":directory,
                "symbol_prefix":prefix,"manifest_sha256":file_identity(path)["sha256"]})
    bundle = {"schema":2,"format":f"{label}-fa4-static-packed","batches":[13,16],
        "entries":entries,
        "geometry":{"sequence":sequence,"heads":heads,"head_dim":head_dim,
            "qkv_stride":qkv_stride,"output_stride":output_stride},
        "generator":file_identity(__file__),
        "validation":"CPU generation/profile/ABI only; GPU numerical and timing acceptance is separate"}
    (root / "bundle.json").write_text(json.dumps(bundle,indent=2)+"\n")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--dependency-root",type=Path,required=True)
    parser.add_argument("--cuda-root",type=Path,required=True)
    parser.add_argument("--profile-label",default="b11",
                        help="Geometry label; also the artifact/bundle name prefix")
    parser.add_argument("--sequence",type=int,default=361)
    parser.add_argument("--heads",type=int,default=12)
    parser.add_argument("--head-dim",type=int,default=32)
    parser.add_argument("--allow-toolchain-mismatch",action="store_true")
    args = parser.parse_args()
    if min(args.sequence,args.heads,args.head_dim) < 1:
        parser.error("sequence, heads and head_dim must be positive")
    root = args.output_dir.resolve()
    deps = args.dependency_root.resolve()
    if root == HERE.parents[2] or root == HERE or HERE in root.parents or root == deps or deps in root.parents:
        parser.error("output must be external to repository root, build helpers and dependency overlay")
    if root.exists() and any(root.iterdir()):
        parser.error("output must be new or empty; refusing to overwrite")
    root.mkdir(parents=True,exist_ok=True)
    for batch in (13,16):
        for mode in ("exact","mask"):
            command = [sys.executable,str(HERE/"generate_fa4.py"),"--mode",mode,
                "--fixed-batch",str(batch),"--dependency-root",str(deps),
                "--output-dir",str(root/f"b{batch}"/mode),"--cuda-root",str(args.cuda_root.resolve()),
                "--profile-label",args.profile_label,"--sequence",str(args.sequence),
                "--heads",str(args.heads),"--head-dim",str(args.head_dim)]
            if args.allow_toolchain_mismatch:
                command.append("--allow-toolchain-mismatch")
            subprocess.run(command,check=True)
    write_bundle(root,args.profile_label,args.heads,args.head_dim,args.sequence)
    print(root/"bundle.json")


if __name__ == "__main__":
    main()
