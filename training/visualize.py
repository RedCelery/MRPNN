"""Render old/fitted CUDA weights and MC reference; preserve linear HDR pixels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--nn-spp", type=int, default=256)
    parser.add_argument("--mc-spp", type=int, default=2048)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/"training/runs/physical-smoke/render-comparison"
    assets = output/"assets"
    assets.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + env["PATH"]
    cases = [("cloud_isotropic","CLOUD03",0.,.5,.2,"云 · 各向同性","训练内体积"),
             ("cloud_forward","CLOUD03",.857,6.,1.4,"云 · 前向散射","训练内体积"),
             ("bunny","BUNNY",.5,1.,2.7,"兔子","整个体积留出")]
    report = dict(width=args.width, nn_spp=args.nn_spp, mc_spp=args.mc_spp,
                  camera=[1.05,.62,1.35], albedo=[.8,.9,.999], seed=20261022,
                  background="black, no environment", tone="ACES followed by gamma 1/2.2",
                  cases=[])
    displays=[]
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    with tempfile.TemporaryDirectory(prefix="mrpnn-render-") as temp:
        for name,scene,g,alpha,angle,title,split in cases:
            results={}
            item=dict(name=name,title=title,split=split,g=g,alpha=alpha,light_angle=angle,
                      images={},render_ms={})
            for mode,exe,spp,reference in (
                ("original","RenderCompare",args.nn_spp,0),
                ("fitted","FittedRenderCompare",args.nn_spp,0),
                ("reference","RenderCompare",args.mc_spp,1)):
                binary=Path(temp)/"image.bin"
                process=subprocess.run([str(root/f"build-5080/Release/{exe}.exe"),
                    str(root/f"TestCase/{scene}"),str(binary),str(args.width),str(spp),
                    str(reference),str(g),str(alpha),str(angle)],cwd=root,env=env,
                    capture_output=True,timeout=600)
                log=process.stdout.decode(errors="replace")+process.stderr.decode(errors="replace")
                if process.returncode:
                    raise RuntimeError(log[-3000:])
                with open(binary,"rb") as f:
                    magic,w,n,stride=np.fromfile(f,"<i4",4)
                    assert magic==0x52434D50 and w==args.width and stride==9
                    array=np.fromfile(f,"<f4").reshape(w,w,9)
                assert np.isfinite(array).all() and (array>=0).all()
                results[mode]=array
                item["render_ms"][mode]=float(re.search(r"render_ms=([\d.]+)",log)[1])
            np.savez_compressed(assets/f"{name}-linear.npz",**results)
            # Browser loads a compact float32 RGB block through a local script.
            # JSON, not a remote dependency or encoded image resource.
            images={}
            for field,start in (("full",0),("indirect",3)):
                images[field]={}
                for mode,array in results.items():
                    x=array[:,:,start:start+3]
                    images[field][mode]=np.round(x,7).reshape(-1).tolist()
                    y=x*2
                    tone=np.clip(y*(2.51*y+.03)/(y*(2.43*y+.59)+.14),0,1)**(1/2.2)
                    path=assets/f"{name}-{field}-{mode}.png"
                    Image.fromarray(np.uint8(tone*255+.5)).save(path)
                    item["images"][f"{field}_{mode}"]=path.name
            foreground=results["reference"][:,:,:3].max(-1)>.002
            item["metrics"]={}
            for field,start in (("full",0),("indirect",3)):
                ref=results["reference"][:,:,start:start+3]
                item["metrics"][field]={}
                for mode in ("original","fitted"):
                    error=results[mode][:,:,start:start+3]-ref
                    item["metrics"][field][mode]=dict(
                        rmse=float(np.sqrt((error[foreground]**2).mean())),
                        mae=float(np.abs(error[foreground]).mean()))
            item["reference_rms_standard_error"]=float(np.sqrt(
                (results["reference"][:,:,6:9][foreground]**2).mean()))
            item["foreground_pixels"]=int(foreground.sum())
            item["data"]=images
            report["cases"].append(item)
            # Export static four-column contact sheet for sharing.
            row=Image.new("RGB",(args.width*4,args.width+36),(16,20,29))
            for j,mode in enumerate(("original","fitted","reference")):
                row.paste(Image.open(assets/f"{name}-full-{mode}.png"),(j*args.width,36))
                ImageDraw.Draw(row).text((j*args.width+10,10),
                    f"{scene} / "+{"original":"Original","fitted":"Fitted","reference":"MC"}[mode],
                    font=font,fill=(235,241,248))
            difference=np.abs(results["fitted"][:,:,:3]-results["original"][:,:,:3]).mean(-1)
            from matplotlib import colormaps
            rgb=(colormaps["magma"](np.clip(difference/.05,0,1))[:,:,:3]*255).astype("uint8")
            row.paste(Image.fromarray(rgb),(args.width*3,36))
            ImageDraw.Draw(row).text((args.width*3+10,10),"Change / max 0.05",font=font,fill=(235,241,248))
            displays.append(row)
            print(title,json.dumps(item["metrics"]),flush=True)
    sheet=Image.new("RGB",(args.width*4,(args.width+36)*len(displays)))
    for i,row in enumerate(displays):
        sheet.paste(row,(0,i*(args.width+36)))
    sheet.save(output/"render-contact-sheet.png")
    data=report
    (assets/"render-data.js").write_text("window.RENDER_DATA="+json.dumps(data,separators=(",",":"))+";",
                                        encoding="utf-8")
    summary={k:v for k,v in report.items() if k!="cases"}
    summary["cases"]=[{k:v for k,v in c.items() if k!="data"} for c in report["cases"]]
    summary["weights_sha256"]=hashlib.sha256(
        (root/"training/runs/physical-smoke/NNWeight.cuh").read_bytes()).hexdigest()
    (output/"render-report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
