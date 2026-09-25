"""Offline mother-series QC for a review gallery; does not assign fate or rewrite lineage IDs."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)

def transient_flags(g, floor=.20, span_max=9):
    """Find short excursions with a return to compatible flanks in >=2 signals.

    Flanks: 3 points each, log-linear baseline between flank median times.
    Seed: centered 11-point log-median residual; >=20% and 4 robust SD.
    Grow to <=9 frames. Require extrapolated flanks to agree within 25%,
    >=2/3 agreeing mass, area, EFD-volume signals on at least 60% of the block.
    Flag only positive excursions; sustained shifts and fate are not classified.
    """
    vals=g[['phase_mass','area_px','volume_um3_efd']].to_numpy(float)
    frames=g.frame.to_numpy(int)
    with np.errstate(invalid='ignore', divide='ignore'):
        z=np.log(np.where(vals>0,vals,np.nan))
    s=pd.DataFrame(z)
    baseline=s.rolling(11,center=True,min_periods=5).median().to_numpy()
    res=z-baseline
    mad=pd.DataFrame(abs(res)).rolling(31,center=True,min_periods=8).median().to_numpy()*1.4826
    # Cap local scale inflation by whole-channel robust first-difference noise.
    dz=np.diff(z,axis=0)
    noise=np.nanmedian(abs(dz-np.nanmedian(dz,axis=0)),axis=0)*1.4826/np.sqrt(2)
    threshold=np.maximum(np.log1p(floor),4*np.minimum(mad,2*noise))
    seeds=np.flatnonzero((abs(res)>threshold).sum(1)>=2)
    marked=np.zeros(len(g),bool)
    segments=[]
    for seed in seeds:
        for length in range(1,span_max+1):
            for start in range(max(3,seed-length+1),min(seed+1,len(g)-length-2)):
                stop=start+length
                if frames[stop+2]-frames[start-3]>length+7:
                    continue
                l=np.nanmedian(z[start-3:start],axis=0)
                r=np.nanmedian(z[stop:stop+3],axis=0)
                ft=frames[start:stop,None]
                lt=np.median(frames[start-3:start]); rt=np.median(frames[stop:stop+3])
                # Compare extrapolated flanks at the SAME time. Comparing their
                # levels alone confuses normal pre-division growth with a spike.
                center=(frames[start]+frames[stop-1])/2
                def slope(a,b):
                    return np.nanmedian(np.array([(z[j]-z[i])/(frames[j]-frames[i])
                        for i in range(a,b) for j in range(i+1,b)]),axis=0)
                ls=slope(start-3,start); rs=slope(stop,stop+3)
                lp=l+ls*(center-lt); rp=r+rs*(center-rt)
                stable=np.abs(rp-lp)<np.log(1.25)
                pred=l+(r-l)*(ft-lt)/(rt-lt)
                delta=z[start:stop]-pred
                th=np.maximum(np.log1p(floor),np.nanmedian(threshold[start:stop],axis=0))
                signs=np.where(delta>th,1,np.where(delta<-th,-1,0))
                pos=((signs>0)&stable).sum(1)>=2
                neg=((signs<0)&stable).sum(1)>=2
                if pos.mean()>=.6:
                    marked[start:stop]|=pos
                    segments.append((int(frames[start]),int(frames[stop-1])))
    return marked

def scan(g, extra=None):
    ok=~g.is_outlier.to_numpy(bool)&~g.touches_border.to_numpy(bool)
    if extra is not None:ok&=~extra
    ok&=np.isfinite(g.phase_mass)&np.isfinite(g.volume_um3_efd)&(g.phase_mass>0)&(g.volume_um3_efd>0)
    v=g.loc[ok].reset_index(drop=True)
    f=v.frame.to_numpy(int); mm=v.phase_mass.to_numpy(); vv=v.volume_um3_efd.to_numpy()
    found=[]
    for i in range(2,len(v)-1):
        if f[i]-f[i-1]>8:continue
        # Abrupt reduction between nearest good measurements, not a gradual trend.
        if not(mm[i]/mm[i-1]<=.78 and vv[i]/vv[i-1]<=.85):continue
        # Estimate each plateau on its own side of the observed edge. The
        # excluded gap is separately limited to 8 frames above.
        bef=np.arange(max(0,i-3),i); bef=bef[f[bef]>=f[i-1]-8]
        aft=np.arange(i,min(len(v),i+3)); aft=aft[f[aft]<=f[i]+8]
        if len(bef)<2 or len(aft)<2:continue
        mr=np.median(mm[aft])/np.median(mm[bef]); vr=np.median(vv[aft])/np.median(vv[bef])
        if not(.25<=mr<=.78 and .25<=vr<=.85 and abs(mr-vr)<=.25):continue
        # First valid post frame; actual division is bracketed across the gap.
        found.append(dict(frame=int(f[i]),pre_frame=int(f[i-1]),mass_ratio=mr,volume_ratio=vr,
                          score=mm[i]/mm[i-1]+vv[i]/vv[i-1]))
    # Only collapse step candidates corresponding to the same local drop, <=3 frames.
    kept=[]
    for r in found:
        if kept and r['frame']-kept[-1]['frame']<=3:
            if r['score']<kept[-1]['score']:kept[-1]=r
        else:kept.append(r)
    return kept

def validate_event(g, frame, extra):
    valid=(~g.is_outlier & ~g.touches_border & ~extra &
           np.isfinite(g.phase_mass) & np.isfinite(g.volume_um3_efd) &
           (g.phase_mass>0) & (g.volume_um3_efd>0))
    good=g.loc[valid]
    pre=good[good.frame.between(frame-8,frame-1)].tail(3)
    post=good[good.frame.between(frame,frame+8)].head(3)
    result=dict(n_pre=len(pre),n_post=len(post),mass_ratio=np.nan,volume_ratio=np.nan)
    if len(pre)<2 or len(post)<2:
        return dict(**result,accepted=False,reason='insufficient_good_points')
    mr=float(post.phase_mass.median()/pre.phase_mass.median())
    vr=float(post.volume_um3_efd.median()/pre.volume_um3_efd.median())
    result.update(mass_ratio=mr,volume_ratio=vr)
    accepted=.25<=mr<=.78 and .25<=vr<=.85 and abs(mr-vr)<=.25
    # Medians alone can borrow a future division and accept a preceding
    # tracker event even though its first good post point is still growing.
    # Use the same local-edge requirement as the missing-step scan.
    if accepted:
        before=pre.iloc[-1];after=post.iloc[0]
        result.update(lower_frame=int(before.frame)+1,upper_frame=int(after.frame))
        if after.frame-before.frame>8:
            return dict(**result,accepted=False,reason='local_gap_too_long')
        if after.phase_mass/before.phase_mass>.78 or after.volume_um3_efd/before.volume_um3_efd>.85:
            return dict(**result,accepted=False,reason='no_local_step_at_candidate')
    return dict(**result,accepted=bool(accepted),reason='mass_volume_step' if accepted else 'no_supported_step')

def review_events(g, original_qc, extra):
    """Revalidate every existing mother event, then add supported missing steps.

    Recovered events are analysis events, never fabricated daughter IDs.
    Timing is bounded by adjacent good measurements; the first good post point
    is used when no existing tracker event lies inside that interval.
    """
    records=[]
    for r in original_qc.itertuples():
        check=validate_event(g,int(r.frame),extra)
        record=dict(frame=int(r.frame),previously_accepted=bool(r.validated),
            origin='tracker',lower_frame=int(r.frame),upper_frame=int(r.frame))
        record.update(check)
        records.append(record)
    # Several tracker candidates inside the same excluded-data gap cannot
    # represent independent observed drops. Keep the latest candidate only;
    # retain the full uncertainty bracket and rejected records for auditing.
    brackets={}
    for r in sorted(records,key=lambda r:r['frame'],reverse=True):
        if not r['accepted']:continue
        key=(r['lower_frame'],r['upper_frame'])
        if key in brackets:
            r.update(accepted=False,reason='duplicate_good_measurement_step')
        else:brackets[key]=r
    for s in scan(g,extra):
        existing=[r for r in records if r['accepted'] and
                  s['pre_frame']<r['frame']<=s['frame']]
        if existing:
            continue
        records.append(dict(frame=s['frame'],previously_accepted=False,origin='recovered',
            lower_frame=s['pre_frame']+1,upper_frame=s['frame'],
            n_pre=np.nan,n_post=np.nan,mass_ratio=s['mass_ratio'],volume_ratio=s['volume_ratio'],
            accepted=True,reason='persistent_step_between_good_measurements'))
    return pd.DataFrame(records,columns=['frame','previously_accepted','origin','lower_frame','upper_frame',
        'n_pre','n_post','mass_ratio','volume_ratio','accepted','reason']).sort_values('frame')
