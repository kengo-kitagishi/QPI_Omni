"""Reconcile the 260517 Omnipose training set to a clean 200-image folder.

Reusable QC/backfill tool. Each time a bad position/channel is found, add it to
EXCLUDE_POSCH or EXCLUDE_POS below and re-run: any offending images already in
the train folder are removed and replaced (same phase) from clean channels, so
the phase distribution and total stay fixed.

Sampling policy (see memory feedback_trainset_starvation_balance):
  - dense in 2% glucose periods (growth P1 + recovery P4), sparse in low-glucose
  - many positions x many channels (ch01-10 only; edge ch00/ch11 excluded)
  - per-frame QC: signal present, no subtraction halo, no horizontal banding
  - drop catalogued drift frames via bad_frames.json
"""
import json, re, glob, random, math
from pathlib import Path
from collections import defaultdict
import numpy as np, tifffile, shutil
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(r"F:\260517\2per_0055per_0per_2per_crop_sub")
OUT  = Path(r"C:\Users\QPI\Desktop\train")
BADJSON = Path(r"F:\260517\drift_session_20260521T2142\bad_frames.json")
random.seed(101)

# phase: (name, start, end_excl, target_count) -- 2% dense, starvation sparse
PHASES = [("P1_2%",2,2019,85),("P2_0055%",2019,2307,15),
          ("P3_0%",2307,2885,25),("P4_2%rec",2885,3748,75)]
CHANNELS = list(range(1,11))                 # ch01..ch10 (exclude ch00, ch11)
EXCLUDE_POS   = {61}                          # whole positions to drop
EXCLUDE_POSCH = {(21,6),(56,8),(61,9),(56,10),(76,10),(81,3)}  # (pos,ch) to drop
CAP = 8                                       # max images per (pos,ch)

BAD = json.loads(BADJSON.read_text(encoding="utf-8"))
NAME = re.compile(r"Pos(\d+)_ch(\d+)_img_(\d+)")

def badset(pos): return set(int(k) for k in BAD.get(f"Pos{pos}",{}).get("bad_timepoints",{}).keys())
def fpath(pos,ch,idx): return ROOT/f"Pos{pos}"/"output_phase"/"channels"/"crop_sub_rawraw"/"z000"/f"ch{ch:02d}"/f"img_{idx:09d}_ph_000_phase.tif"
def load(pos,ch,idx):
    p=fpath(pos,ch,idx)
    if not p.exists(): return None
    try: return tifffile.imread(p).astype(np.float32)
    except Exception: return None
def metrics(im):
    H,W=im.shape
    sig=np.percentile(im,99)-np.median(im)
    halo=im[:,int(W*0.8):].mean()-im[:,:int(W*0.6)].mean()   # right minus left (subtraction wash)
    band=np.var(im.mean(axis=1))                              # horizontal banding (drift streaks)
    med=np.median(im); mad=np.median(np.abs(im-med))+1e-6
    # fraction of cell-bright pixels in the channel interior (drop left wall 8px,
    # right open end 20%); the bright trap-wall bar inflates p99/sig even when the
    # trap is empty, so frac on the interior is the reliable cell-presence gate.
    frac=float(np.mean((im[:, 8:int(W*0.8)] > med+5*mad)))
    return sig,halo,band,frac
def good(im):
    if im is None: return False
    sig,halo,band,frac=metrics(im)
    return frac>0.02 and halo<0.9 and band<0.25 and im.std()<2.5
def populated(pos,ch):
    im=load(pos,ch,1500)
    if im is None: return False
    sig,halo,band,frac=metrics(im)
    return im.std()>0.22 and frac>0.02 and band<0.25
def phase_of(idx):
    for i,(_,s,e,_n) in enumerate(PHASES):
        if s<=idx<e: return i
    return 3
def excluded(pos,ch): return pos in EXCLUDE_POS or (pos,ch) in EXCLUDE_POSCH

def main():
    OUT.mkdir(exist_ok=True)
    # 1. remove current images that violate exclusions or are catalogued bad frames
    removed=0
    for f in glob.glob(str(OUT/"Pos*_img_*.tif")):
        m=NAME.search(Path(f).name); pos,ch,idx=int(m.group(1)),int(m.group(2)),int(m.group(3))
        if excluded(pos,ch) or idx in badset(pos) or ch not in CHANNELS:
            Path(f).unlink(); removed+=1
    print(f"removed {removed} offending images")

    # 2. current inventory
    existing=defaultdict(set)
    for f in glob.glob(str(OUT/"Pos*_img_*.tif")):
        m=NAME.search(Path(f).name); existing[(int(m.group(1)),int(m.group(2)))].add(int(m.group(3)))
    used=lambda k: len(existing.get(k,()))
    have=[0,0,0,0]
    for k,idxs in existing.items():
        for idx in idxs: have[phase_of(idx)]+=1

    # 3. candidate clean channels
    cands=[]
    for pos in range(1,105):
        for c in CHANNELS:
            if excluded(pos,c): continue
            if populated(pos,c): cands.append((pos,c))
    random.shuffle(cands); cands.sort(key=lambda k: used(k))
    print(f"clean candidate channels: {len(cands)}")

    # 4. backfill each phase up to target
    for ph,(_n,s,e,tgt) in enumerate(PHASES):
        deficit=tgt-have[ph]
        got=0
        for (pos,c) in cands:
            if got>=deficit: break
            if used((pos,c))>=CAP: continue
            for x in random.sample(range(s+50,e-1), min(80,e-1-(s+50))):
                if x in badset(pos) or x in existing[(pos,c)]: continue
                if good(load(pos,c,x)):
                    shutil.copy2(fpath(pos,c,x), OUT/f"Pos{pos}_ch{c:02d}_img_{x:09d}.tif")
                    existing[(pos,c)].add(x); got+=1; break
        if deficit>0: print(f"{PHASES[ph][0]}: backfilled {got}/{deficit}")

    # 5. verify + contact sheet
    allf=sorted(glob.glob(str(OUT/"Pos*_img_*.tif")))
    hist=[0,0,0,0]; contam=0; poss=set()
    for f in allf:
        m=NAME.search(Path(f).name); pos,idx=int(m.group(1)),int(m.group(3))
        hist[phase_of(idx)]+=1; poss.add(pos)
        if idx in badset(pos): contam+=1
    print(f"TOTAL {len(allf)} | phase {hist} | bad-contam {contam} | positions {len(poss)}")

    cols=10; rows=math.ceil(len(allf)/cols)
    fig,axs=plt.subplots(rows,cols,figsize=(cols*2.0,rows*0.62)); axs=np.array(axs).reshape(-1)
    for ax in axs: ax.axis("off")
    for ax,f in zip(axs,allf):
        im=tifffile.imread(f).astype(np.float32)
        ax.imshow(im,cmap="gray",vmin=np.percentile(im,1),vmax=np.percentile(im,99.5))
        m=NAME.search(Path(f).name); ax.set_title(f"P{m.group(1)}c{m.group(2)}_{int(m.group(3))}",fontsize=4.2,pad=0.8)
    plt.subplots_adjust(left=0.005,right=0.995,top=0.985,bottom=0.005,wspace=0.05,hspace=0.55)
    sheet=r"C:\Users\QPI\Desktop\_train200_contact_sheet.png"; plt.savefig(sheet,dpi=200)
    print("contact sheet:",sheet)

if __name__=="__main__":
    main()
