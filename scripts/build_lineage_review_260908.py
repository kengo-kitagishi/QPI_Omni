"""Build a reversible review package from 260908 CSVs and user channel choices.

Recomputes only mother-series QC / cycle fits. Segmentation, lineage IDs and
biological fate are unchanged. CSVs in the source tree are never overwritten.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,re,sys
from html import escape
import numpy as np
import pandas as pd
from PIL import Image
import lineage_review_qc as review

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def annotation_notes(path):
    if path is None:return {}
    notes={}
    for line in path.read_text(encoding='utf8').splitlines():
        match=re.match(r'(Pos\d+) (ch\d+):\s*(.*)',line)
        if match:notes[(match[1],match[2])]=match[3]
    return notes

def fit_cycles(g,events):
    frames=g.frame.to_numpy();mass=g.phase_mass.to_numpy()
    valid=g.review_valid.to_numpy(); times=(frames-2)/12
    fits=[]
    for start,end in zip(events[:-1],events[1:]):
        use=valid&(frames>=start)&(frames<end)&np.isfinite(mass)&(mass>0)
        if use.sum()<5:continue
        x=times[use];y=np.log(mass[use]);slope,intercept=np.polyfit(x,y,1)
        ss=float(np.sum((y-y.mean())**2));res=float(np.sum((y-(intercept+slope*x))**2))
        fits.append(dict(start_frame=int(start),end_frame=int(end),t_start=(start-2)/12,
            t_end=(end-2)/12,n_points=int(use.sum()),slope_per_h=slope,intercept=intercept,
            doubling_time_h=np.log(2)/slope if slope>0 else np.nan,r2=1-res/ss if ss>0 else np.nan))
    return fits

def prepare(args,out):
    csv=args.root/'_lineage_consolidated/all_cells_lineage_data3D.csv.gz'
    qcfile=args.root/'_lineage_consolidated/all_cells_divisions_qc.csv.gz'
    choices=pd.read_csv(args.audit/'channels.csv').set_index(['pos','ch'])
    latest=annotation_notes(args.annotations)
    allrows=pd.read_csv(csv);m=allrows[allrows.cell_id==0].copy()
    q=pd.read_csv(qcfile);q=q[q.parent_id==0]
    factor=.658*.34567514677103717**2/(2*np.pi*.00018)*1e-3
    m['phase_mass']=m.total_phase*factor;m['t']=(m.frame-2)/12
    frames=[];events=[];channels=[]
    keys=sorted(m.groupby(['pos','ch']).groups,key=lambda k:(int(k[0][3:]),k[1]))
    for i,(pos,ch) in enumerate(keys,1):
        g=m[(m.pos==pos)&(m.ch==ch)].sort_values('frame').reset_index(drop=True)
        note='';cutoff=np.nan;excluded=False
        if (pos,ch) in choices.index:
            choice=choices.loc[(pos,ch)];note=choice.note;excluded=bool(choice.exclude);cutoff=choice.cutoff_h
        if f'{pos}/{ch}' in args.exclude_channel:
            excluded=True;note=latest.get((pos,ch),note)
        if excluded:
            channels.append(dict(pos=pos,ch=ch,included=False,cutoff_h=cutoff,note=note))
            continue
        g=g[g.t<=args.end_hour].reset_index(drop=True)
        if np.isfinite(cutoff):g=g[g.t<cutoff].reset_index(drop=True)
        extra=review.transient_flags(g)
        g['review_outlier_added']=extra&~g.is_outlier&~g.touches_border
        g['review_valid']=~g.is_outlier&~g.touches_border&~extra&np.isfinite(g.phase_mass)&np.isfinite(g.volume_um3_efd)
        original=q[(q.pos==pos)&(q.ch==ch)&q.frame.between(g.frame.min(),g.frame.max())]
        ev=review.review_events(g,original,extra);ev['pos']=pos;ev['ch']=ch
        old=set(original.loc[original.validated,'frame'].astype(int))
        new=set(ev.loc[ev.accepted,'frame'].astype(int))
        channels.append(dict(pos=pos,ch=ch,included=True,cutoff_h=cutoff,note=note,
            old_divisions=len(old),review_divisions=len(new),added_divisions=len(new-old),
            removed_divisions=len(old-new),added_outlier_points=int(g.review_outlier_added.sum()),
            n_frames=len(g),valid_points=int(g.review_valid.sum())))
        frames.append(g);events.append(ev)
        print(f'QC {i}/{len(keys)} {pos} {ch}: outliers +{g.review_outlier_added.sum()}, divisions {len(old)} -> {len(new)}',flush=True)
    f=pd.concat(frames,ignore_index=True);e=pd.concat(events,ignore_index=True);c=pd.DataFrame(channels)
    f.to_csv(out/'mother_measurements_review.csv.gz',index=False)
    e.to_csv(out/'division_events_review.csv',index=False,encoding='utf-8-sig')
    c.to_csv(out/'channel_review.csv',index=False,encoding='utf-8-sig')
    manifest=dict(dataset='260908_outside_quad',status='review_only',fate='manual; not inferred',
        source_csv=str(csv),source_sha256=digest(csv),source_qc_sha256=digest(qcfile),
        choices_sha256=digest(args.audit/'channels.csv'),qc_code_sha256=digest(Path(review.__file__)),
        end_hour=args.end_hour,end_frame=int(round(2+args.end_hour*12)),
        n_channels=int(c.included.sum()),excluded_channels=c.loc[~c.included,['pos','ch']].to_dict('records'),
        missing_channels=['Pos6 ch00','Pos6 ch01'],cutoff={'Pos85 ch07':'time_h < 90'},
        outlier_rule='positive transient 1-9 frames; >=20% and 4 robust SD; >=2/3 mass-area-EFD; extrapolated flanks agree within 25%; >=60% of block',
        division_rule='all candidates tested; nearest <=3 good points per side within 8 frames; >=2 each; mass ratio .25-.78, volume .25-.85, abs difference <=.25; persistent missing steps added',
        source_data_note='original physical columns preserved; review_valid is the mask used for fits',
        added_outlier_points=int(c.added_outlier_points.sum()),added_divisions=int(c.added_divisions.sum()),
        removed_divisions=int(c.removed_divisions.sum()))
    manifest['extra_exclusions']=args.exclude_channel
    manifest['annotations_sha256']=digest(args.annotations) if args.annotations else None
    manifest['division_rule']+='; require local mass/volume edge for tracker events, gap <=8 frames; deduplicate identical good-point brackets; scan pre plateau within8 frames of last good pre point'
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    return f,e,c,manifest

def round_report(args,out,f,e,c):
    """Audit second-round marks in actual axis hours (no legacy offset)."""
    notes=annotation_notes(args.annotations)
    if not notes:return
    lines=['# 今回の指摘のCSV照合','',
        '生画像全フレームの目視確認ではありません。時刻は今回のHTMLの軸座標をそのまま使用。',
        'outlierの閾値は変更していません。採用点と表示される×（既に除外）を区別してください。',
        '分裂は局所の減少と重複を再検査。欠測区間の正確な分裂時刻、残存outlier、死亡は自動確定しません。','']
    rows=[];windows=[]
    for (pos,ch),note in notes.items():
        lines.extend([f'## {pos} {ch}','',note,''])
        g=f[(f.pos==pos)&(f.ch==ch)];ev=e[(e.pos==pos)&(e.ch==ch)].copy()
        if g.empty:lines.extend(['指定により解析対象外。','']);continue
        lines.extend(['|指摘 h|±0.35 h内の分裂候補（採否）|±0.15 h内の有効点 / 除外点|','|---:|---|---|'])
        for t in sorted(set(map(float,re.findall(r'@([\d.]+)h',note)))):
            near=ev[((ev.frame-2)/12).between(t-.35,t+.35)]
            points=g[g.t.between(t-.15,t+.15)]
            label='; '.join(f'{(r.frame-2)/12:.3f} h: '+('採用' if r.accepted else '非採用')+f' ({r.reason})' for r in near.itertuples()) or '該当候補なし'
            rows.append(dict(pos=pos,ch=ch,mark_h=t,events=label,valid_points=int(points.review_valid.sum()),excluded_points=int((~points.review_valid).sum())))
            lines.append(f'|{t:.1f}|{label}|{points.review_valid.sum()} / {(~points.review_valid).sum()}|')
            w=g[g.t.between(t-.75,t+.75)].copy();w['mark_h']=t;windows.append(w)
        lines.append('')
    pd.DataFrame(rows).to_csv(out/'round2_audit.csv',index=False,encoding='utf-8-sig')
    pd.concat(windows,ignore_index=True).to_csv(out/'round2_windows.csv.gz',index=False)
    (out/'round2_review.md').write_text('\n'.join(lines),encoding='utf8')
    (out/'annotations_round2.txt').write_text(args.annotations.read_text(encoding='utf8'),encoding='utf8')

def make_report(args,out,c,e):
    a=pd.read_csv(args.audit/'event_audit.csv').fillna('')
    raw=pd.read_csv(args.audit/'channels.csv').fillna('')
    lines=['# 260908 指摘箇所の確認記録','',
        'CSVと現行コードを全指摘に照合した記録。生画像を全フレーム目視したものではない。',
        '274マーク（重複込み）、251件の異なるマーク、92系列。死亡の記述はユーザーの判断として保持し、自動分類しない。',
        f'今回の表示・解析窓は0–{args.end_hour:g} h。Pos85/ch07は90 h未満。窓外の指摘は記録のみ保持する。',
        '旧HTMLはbbox_inches="tight"とクリック座標の不一致がある。補正時刻は画像の軸位置から推定した値で、クリックの精度を保証しない。0.0 hは左端への丸めがあり逆変換できない。',
        '新HTMLは画像の余白を切り詰めず保存するためクリックと軸の座標が一致する。','']
    for r in raw.itertuples():
        lines.extend([f'## {r.pos} {r.ch}','',r.note,''])
        sub=c[(c.pos==r.pos)&(c.ch==r.ch)]
        if len(sub):
            s=sub.iloc[0]
            if not s.included:lines.extend(['指定により解析対象外。元データは保持。',''])
            else:lines.extend([f"確認版: 分裂 {s.old_divisions:.0f} → {s.review_divisions:.0f}、追加outlier {s.added_outlier_points:.0f}点。",''])
        lines.extend(['|マーク|旧クリック h|補正推定 h|照合結果|','|---|---:|---:|---|'])
        for z in a[(a.pos==r.pos)&(a.ch==r.ch)].itertuples():
            ev=e[(e.pos==r.pos)&(e.ch==r.ch)&e.accepted]
            near=ev[((ev.frame-2)/12).between(z.window_start_h,z.window_end_h)]
            if z.kind=='分裂見逃し':
                detail=('確認版の分裂: '+','.join(f'f{int(f)} ({(f-2)/12:.2f} h)' for f in near.frame)) if len(near) else '±0.4 h内は未対応。時刻ずれ・元画像の個別確認が必要'
            else:
                detail='追加outlier候補: '+(str(z.new_outlier_frames) or 'なし。死亡/メモの意味は自動確定しない')
            if len(sub) and not sub.iloc[0].included:
                detail='系列全体を指定により解析対象外'
            elif z.corrected_h>args.end_hour or (r.pos=='Pos85' and r.ch=='ch07' and z.corrected_h>=90):
                detail='打切り後のため今回の解析窓外（記録のみ保持）'
            lines.append(f'|{z.kind}|{z.click_h:.1f}|{z.corrected_h:.2f}|{detail}|')
        lines.append('')
    (out/'annotation_review.md').write_text('\n'.join(lines),encoding='utf8')

def render(args,out,f,e,c,manifest):
    os.environ['QPI_FIGURE_INBOX_ROOT']=str(out/'figure_inbox')
    import lineage_html_gallery_260517 as gallery
    gallery.END_FRAME=int(manifest['end_frame'])
    gallery.PLOT_RIGHT=.98  # Leave room for the endpoint tick label at 100 h.
    oldhtml=args.root/'_qc/lineage_html/lineage_gallery_csv_20260921T112710.html'
    old=oldhtml.read_text(encoding='utf8')
    style=re.search(r'<style>(.*?)</style>',old,re.S).group(1)
    toolbar=re.search(r"(<div id='bar'>.*?</div>)<h1",old,re.S).group(1)
    js=re.search(r'(<script>.*?</script>)',old,re.S).group(1)
    js=re.sub(r'const ENDH=[0-9.]+,',f'const ENDH={manifest["end_hour"]:.5f},',js,count=1)
    js=re.sub(r',R=[0-9.]+;',f',R={gallery.PLOT_RIGHT:.5f};',js,count=1)
    js=re.sub(r'const KEY=.*?;let MODE=',f'const KEY="lineage_review_260908_{out.name}";let MODE=',js,count=1)
    parts=[];fits=[]
    latest=annotation_notes(args.annotations)
    oldparams=json.loads((oldhtml.parent/'params_20260921T112710.json').read_text())
    limits=oldparams['ylims']
    included=c[c.included]
    for i,r in enumerate(included.itertuples(),1):
        g=f[(f.pos==r.pos)&(f.ch==r.ch)].sort_values('frame')
        ev=e[(e.pos==r.pos)&(e.ch==r.ch)]
        oldevents=set(ev.loc[ev.previously_accepted,'frame'].astype(int))
        newevents=set(ev.loc[ev.accepted,'frame'].astype(int))
        divs=np.array(sorted(newevents),int)
        fits_one=fit_cycles(g,divs)
        fits.extend(dict(pos=r.pos,ch=r.ch,**row) for row in fits_one)
        d=dict(t=g.t.to_numpy(),frame=g.frame.to_numpy(),ri=g.mean_ri.to_numpy(),
            mass=g.phase_mass.to_numpy(),vol=g.volume_um3_efd.to_numpy(),
            valid=g.review_valid.to_numpy(),outl=(g.is_outlier|g.review_outlier_added).to_numpy(),
            outl_added=g.review_outlier_added.to_numpy(),bord=g.touches_border.to_numpy(),
            conc=g.density_pg_um3_efd.to_numpy()*1000,
            bad_t=np.array([]),bad_mass=np.array([]),bad_vol=np.array([]),bad_ri=np.array([]),
            div_all=np.sort(ev.frame.unique()),div_ok=divs,div_added=np.array(sorted(newevents-oldevents),int),
            div_removed=np.array(sorted(oldevents-newevents),int),show_removed=False,
            fits=fits_one,vol_col='volume_um3_efd',mass_col_qc='phase_mass')
        png=gallery.render(r.pos,r.ch,d,limits,provenance=dict(dataset='260908_review',source='review CSV',
            csv=str(out/'mother_measurements_review.csv.gz'),qc_manifest=str(out/'manifest.json')))
        imagepath=out/f'{r.pos}_{r.ch}.png';imagepath.write_bytes(png)
        with Image.open(imagepath) as im:assert im.size==(4500,1860),im.size
        key=f'{r.pos} {r.ch}';anchor=f'{r.pos}_{r.ch}'
        oldpng=oldhtml.parent/f'{r.pos}_{r.ch}_20260921T112710.png'
        comparison=(f"<details><summary>修正前の図を比較</summary><img loading='lazy' src='{escape(os.path.relpath(oldpng,out).replace(os.sep,'/'),quote=True)}'></details>" if oldpng.exists() else '')
        note=str(r.note) if pd.notna(r.note) else ''
        if (r.pos,r.ch) in latest:note+='\n今回の指摘（未解決項目を含む）: '+latest[(r.pos,r.ch)]
        table=ev[['frame','lower_frame','upper_frame','accepted','reason']].copy()
        for col in ['frame','lower_frame','upper_frame']:table[col]=((table[col]-2)/12).round(3)
        table['accepted']=table.accepted.map({True:'採用（fit境界）',False:'非採用（fit不使用）'})
        table.columns=['候補 h','区間下限 h','区間上限 h','判定','理由']
        event_details='<details><summary>分裂候補の採否・欠測区間を確認（非採用は図に描画しません）</summary>'+table.to_html(index=False,escape=True)+'</details>'
        parts.append(f"<section id='{anchor}'><h2>{key} — 分裂 {r.old_divisions:.0f} → {r.review_divisions:.0f} / 追加outlier {r.added_outlier_points:.0f}点</h2>"
            f"<p class='note'>{escape(note)}</p><div class='imgwrap' data-key='{key}' data-divs='{','.join(f'{(v-2)/12:.5f}' for v in divs)}'>"
            f"<img loading='lazy' src='data:image/png;base64,{base64.b64encode(png).decode()}' alt='{key}'></div>"
            f"<div class='cmtwrap'><div class='marks' data-key='{key}'></div><textarea class='cmt' data-key='{key}' placeholder='{key} 確認メモ'></textarea></div>{event_details}{comparison}</section>")
        print(f'FIGURE {i}/{len(included)} {key}',flush=True)
    pd.DataFrame(fits).to_csv(out/'cycle_fits_review.csv',index=False,encoding='utf-8-sig')
    toc=' | '.join(f"<a href='#{r.pos}_{r.ch}'>{r.pos} {r.ch}</a>" for r in included.itertuples())
    excluded=', '.join(f'{r.pos}/{r.ch}' for r in c[~c.included].itertuples())
    heading=(f'<h1>260908 RI / mass / volume — 判定修正の確認版（{len(included)} ch）</h1>'
        '<p><strong>縦線は採用した分裂だけです。不採用の紫の点線は表示しません。</strong>採否と時刻の不確実区間は各系列の表で確認できます。'
        '死亡判定は手動です。緑の縦線＝初回QCから追加した分裂候補、紫×＝追加除外したoutlier。'
        '灰色の縦線＝継続採用した分裂、赤×＝元のoutlier、赤い曲線＝有効点から再計算したcycle fit。追加分裂は系譜IDを作り替えない解析用候補です。</p>'
        '<p>各系列の「修正前の図を比較」で前の図を展開できます。右上からマークを選び、画像をクリックしてコメントを記録できます。'
        '今回のクリック時刻は軸位置と一致するよう修正済みです。旧メモは本文に保持し、新しいメモは別保存です。</p>'
        f'<p>全系列を{manifest["end_hour"]:g} hまでで打ち切り。指定による除外：{excluded}。Pos85/ch07は90 h未満。Pos6/ch00・ch01はBG不足・計測なし。</p>'
        f'<p>追加outlier {manifest["added_outlier_points"]}点、追加分裂 {manifest["added_divisions"]}件、以前の分裂の不採用・保留 {manifest["removed_divisions"]}件。'
        '改善率・正解率を意味する数字ではありません。<a href="annotation_review.md">指摘別の照合記録</a></p>')
    if latest:heading+='<p><a href="round2_review.md">今回の指摘の照合記録</a>：outlier閾値は前版のままです。残存outlier・分裂見逃しは未解決項目を含みます。</p>'
    path=out/'lineage_gallery_review.html'
    path.write_text(f"<!doctype html><html lang='ja'><meta charset='utf-8'><title>260908 判定修正 確認版</title><style>{style}h1{{font-size:20px;margin-top:60px}}details{{margin:12px 0}}section{{scroll-margin-top:55px}}.imgwrap img{{border:0}}</style><body>{toolbar}{heading}<p class='toc'>{toc}</p>{''.join(parts)}{js}</body></html>",encoding='utf8')
    print('HTML:',path,flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('E:/260908_outside_quad/seg'))
    ap.add_argument('--audit',type=Path,default=Path('E:/260908_outside_quad/seg/_qc/review_20260925'))
    ap.add_argument('--out',type=Path,default=Path('E:/260908_outside_quad/seg/_qc/review_20260925/gallery_v1'))
    ap.add_argument('--prepared',action='store_true')
    ap.add_argument('--end-hour',type=float,default=100.0,help='Inclusive end time for this 260908 review')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--annotations',type=Path,help='Additional user annotations; times are current axis hours')
    ap.add_argument('--exclude-channel',action='append',default=[],help='Explicit user choice: PosN/chNN')
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if args.prepared:
        f=pd.read_csv(out/'mother_measurements_review.csv.gz');e=pd.read_csv(out/'division_events_review.csv')
        c=pd.read_csv(out/'channel_review.csv');manifest=json.loads((out/'manifest.json').read_text(encoding='utf8'))
        assert manifest['qc_code_sha256']==digest(Path(review.__file__)),'QC source changed; prepare again'
        assert manifest['end_hour']==args.end_hour,'Cutoff changed; prepare again'
    else:f,e,c,manifest=prepare(args,out)
    make_report(args,out,c,e)
    round_report(args,out,f,e,c)
    if not args.prepare_only:render(args,out,f,e,c,manifest)

if __name__=='__main__':main()
