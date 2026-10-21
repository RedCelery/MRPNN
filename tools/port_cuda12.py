"""One-time migration of legacy CUDA references to texture/surface objects."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for name in ("core/renderResources.cuh", "core/renderResources.cu", "core/volume.cu"):
    path = root / name
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"extern texture<[^;]+>\s+(\w+);",
                    r"extern __device__ cudaTextureObject_t \1;", source)
    source = re.sub(r"(?m)^texture<[^;]+>\s+(\w+);",
                    r"__device__ cudaTextureObject_t \1 = 0;", source)
    source = re.sub(r"(?m)^surface<[^;]+>\s+(\w+);",
                    r"__device__ cudaSurfaceObject_t \1 = 0;", source)
    if name == "core/volume.cu":
        source = '#include "cuda_objects.hpp"\n' + source
        for symbol, array, mode in (
            ("_HGLut", "hglut_dev", "Clamp"),
            ("_DensityVolume", "datas_dev", "Border"),
            ("texRef", "mips_dev[tr_mip + 1]", "Border"),
            ("_HDRI", "env_tex_dev", "Wrap"),
        ):
            pattern = re.escape(symbol) + r"\.normalized = true;.*?cudaBindTextureToArray\(" + re.escape(symbol) + r",[^;]+;"
            source, count = re.subn(pattern,
                f"BindTextureObject({symbol}, {array}, cudaAddressMode{mode});",
                source, flags=re.S)
            assert count == 1, (symbol, count)
        source, count = re.subn(
            r"#define BindMip\(i\).*?cudaBindTextureToArray\(Mip\(i\),[^;]+;",
            "#define BindMip(i) BindTextureObject(Mip(i), mips_dev[i], cudaAddressModeBorder);",
            source, flags=re.S)
        assert count == 1
        source, count = re.subn(
            r"#define BindTR_Mip\(i\).*?cudaBindTextureToArray\(TR_Mip\(i\),[^;]+;",
            "#define BindTR_Mip(i) BindTextureObject(TR_Mip(i), tr_mips_dev[i], cudaAddressModeClamp);",
            source, flags=re.S)
        assert count == 1
        source = source.replace("cudaBindSurfaceToArray(", "BindSurfaceObject(")
    path.write_text(source, encoding="utf-8")
