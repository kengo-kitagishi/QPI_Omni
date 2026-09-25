"""Scientific-behavior regressions; synthetic data, no experiment files required."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lineage_review_qc import transient_flags,scan,validate_event,review_events

def series(values):
    v=np.asarray(values,float)
    return pd.DataFrame(dict(frame=np.arange(len(v)),phase_mass=v,area_px=10*v,
        volume_um3_efd=4*v,is_outlier=False,touches_border=False))

class ReviewQC(unittest.TestCase):
    def test_future_division_does_not_validate_preceding_tracker_candidate(self):
        v=20*np.exp(np.arange(60)*.025);v[30:]*=.5
        g=series(v);extra=np.zeros(len(g),bool)
        self.assertFalse(validate_event(g,29,extra)['accepted'])
        self.assertTrue(validate_event(g,30,extra)['accepted'])

    def test_duplicate_candidates_in_same_gap_are_one_event(self):
        v=20*np.exp(np.arange(60)*.025);v[30:]*=.5
        g=series(v);g.loc[29:31,'is_outlier']=True
        q=pd.DataFrame(dict(frame=[29,31],validated=[True,True]))
        e=review_events(g,q,np.zeros(len(g),bool))
        self.assertEqual(e.loc[e.accepted,'frame'].tolist(),[31])
        self.assertEqual(e.loc[e.accepted,['lower_frame','upper_frame']].values.tolist(),[[29,32]])

    def test_tracker_event_cannot_bridge_more_than_eight_frames(self):
        v=20*np.exp(np.arange(60)*.025);v[30:]*=.5
        g=series(v);g.loc[26:34,'is_outlier']=True
        self.assertFalse(validate_event(g,30,np.zeros(len(g),bool))['accepted'])

    def test_sparse_pre_plateau_locates_drop_after_last_growing_point(self):
        g=series(np.r_[np.full(30,20.),np.full(30,10.)])
        g.loc[21:27,'is_outlier']=True
        g.loc[29:32,'is_outlier']=True
        q=pd.DataFrame(dict(frame=[27,28],validated=[True,False]))
        e=review_events(g,q,np.zeros(len(g),bool))
        self.assertEqual(e.loc[e.accepted,'frame'].tolist(),[33])
        self.assertEqual(e.loc[e.accepted,['lower_frame','upper_frame']].values.tolist(),[[29,33]])

    def test_smooth_growth_is_not_outlier_or_division(self):
        g=series(20*np.exp(np.arange(60)*.025))
        self.assertFalse(transient_flags(g).any())
        self.assertEqual(scan(g),[])

    def test_real_division_preserved(self):
        v=20*np.exp(np.arange(60)*.025);v[30:]*=.5
        g=series(v); extra=transient_flags(g)
        self.assertFalse(extra.any())
        self.assertEqual([r['frame'] for r in scan(g,extra)],[30])

    def test_four_frame_merge_excursion_removed_not_called_division(self):
        v=20*np.exp(np.arange(60)*.025);v[25:29]*=2
        g=series(v);extra=transient_flags(g)
        self.assertTrue(extra[25:29].all())
        self.assertEqual(int(extra.sum()),4)
        self.assertFalse(validate_event(g,29,extra)['accepted'])

    def test_outlier_gap_does_not_hide_real_division(self):
        v=20*np.exp(np.arange(60)*.025);v[30:]*=.5
        g=series(v);g.loc[29:31,'is_outlier']=True
        d=scan(g)
        self.assertEqual(len(d),1)
        self.assertEqual((d[0]['pre_frame'],d[0]['frame']),(28,32))

    def test_sustained_size_increase_not_transient_outlier(self):
        v=20*np.exp(np.arange(60)*.005);v[30:]*=1.7
        self.assertFalse(transient_flags(series(v)).any())

    def test_long_missing_gap_not_recovered(self):
        v=20*np.exp(np.arange(60)*.025);v[30:]*=.5
        g=series(v);g.loc[25:35,'is_outlier']=True
        self.assertEqual(scan(g),[])

if __name__=='__main__':unittest.main()
