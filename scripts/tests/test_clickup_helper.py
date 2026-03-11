import importlib.util
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


MODULE_PATH = Path('/Users/kitak/QPI_Omni/scripts/clickup_helper.py')
SPEC = importlib.util.spec_from_file_location('clickup_helper', MODULE_PATH)
clickup_helper = importlib.util.module_from_spec(SPEC)
sys.modules['clickup_helper'] = clickup_helper
SPEC.loader.exec_module(clickup_helper)


class ClickUpHelperLogicTests(unittest.TestCase):
    def test_parse_duration_from_text(self):
        self.assertEqual(clickup_helper.parse_duration_from_text('2h task'), 120)
        self.assertEqual(clickup_helper.parse_duration_from_text('1時間30分'), 90)
        self.assertEqual(clickup_helper.parse_duration_from_text('45min deep work'), 45)
        self.assertIsNone(clickup_helper.parse_duration_from_text('someday maybe'))

    def test_choose_figure_hub_config_prefers_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.local.json').write_text('{}', encoding='utf-8')
            (root / 'config.json').write_text('{}', encoding='utf-8')
            chosen = clickup_helper.choose_figure_hub_config(root)
            self.assertEqual(chosen, (root / 'config.json').resolve())

    def test_build_figfix_clickup_sync_cmd(self):
        script = Path('/tmp/figure_hub.py')
        config = Path('/tmp/config.json')
        preview = clickup_helper.build_figfix_clickup_sync_cmd(
            script_path=script,
            config_path=config,
            max_tasks=5,
            due_in_days=7,
            list_key='slide',
            only_fig_id='fig_x',
            apply=False,
        )
        self.assertIn('clickup-sync', preview)
        self.assertIn('--list-key', preview)
        self.assertIn('slide', preview)
        self.assertNotIn('--apply', preview)

        apply_cmd = clickup_helper.build_figfix_clickup_sync_cmd(
            script_path=script,
            config_path=config,
            max_tasks=0,
            due_in_days=-3,
            apply=True,
        )
        self.assertIn('--apply', apply_cmd)
        self.assertEqual(apply_cmd[apply_cmd.index('--max-tasks') + 1], '1')
        self.assertEqual(apply_cmd[apply_cmd.index('--due-in-days') + 1], '0')

    def test_merge_intervals(self):
        tz = ZoneInfo('Asia/Tokyo')
        intervals = [
            (datetime(2026, 3, 6, 10, 0, tzinfo=tz), datetime(2026, 3, 6, 11, 0, tzinfo=tz)),
            (datetime(2026, 3, 6, 10, 30, tzinfo=tz), datetime(2026, 3, 6, 12, 0, tzinfo=tz)),
            (datetime(2026, 3, 6, 13, 0, tzinfo=tz), datetime(2026, 3, 6, 14, 0, tzinfo=tz)),
        ]
        merged = clickup_helper.merge_intervals(intervals)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0][0].hour, 10)
        self.assertEqual(merged[0][1].hour, 12)

    def test_subtract_intervals(self):
        tz = ZoneInfo('Asia/Tokyo')
        base = (datetime(2026, 3, 6, 9, 0, tzinfo=tz), datetime(2026, 3, 6, 12, 0, tzinfo=tz))
        busy = [
            (datetime(2026, 3, 6, 10, 0, tzinfo=tz), datetime(2026, 3, 6, 11, 0, tzinfo=tz)),
        ]
        free = clickup_helper.subtract_intervals(base, busy)
        self.assertEqual(len(free), 2)
        self.assertEqual(free[0][0].hour, 9)
        self.assertEqual(free[0][1].hour, 10)
        self.assertEqual(free[1][0].hour, 11)
        self.assertEqual(free[1][1].hour, 12)

    def test_find_free_slots(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=12,
            min_gap_minutes=0,
            planning_days=1,
            max_candidates=3,
        )
        busy_task = clickup_helper.TaskWindow(
            task_id='t1',
            name='busy',
            list_id='x',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 6, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 6, 11, 0, tzinfo=tz),
            due_at=datetime(2030, 3, 6, 11, 0, tzinfo=tz),
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[busy_task],
            scheduled_tasks=[busy_task],
            unscheduled_tasks=[],
            warnings=[],
        )
        slots = clickup_helper.find_free_slots(
            snap,
            cfg=cfg,
            duration_minutes=60,
            list_key='experiment',
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_slots=10,
        )
        self.assertGreaterEqual(len(slots), 1)
        self.assertEqual(slots[0].slot_start.hour, 9)
        self.assertEqual(slots[0].slot_end.hour, 10)

    def test_find_free_slots_skips_competition_day(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=12,
            min_gap_minutes=0,
            no_schedule_day_list_keys=['competition'],
            no_schedule_day_keyword_patterns=['大会'],
        )
        comp_task = clickup_helper.TaskWindow(
            task_id='cmp1',
            name='春季OP 大会',
            list_id='lc',
            list_key='competition',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 6, 4, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 6, 4, 0, tzinfo=tz),
            due_at=datetime(2030, 3, 6, 4, 0, tzinfo=tz),
            duration_minutes=0,
            movable=False,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 7),
            timezone='Asia/Tokyo',
            fetched_tasks=[comp_task],
            scheduled_tasks=[comp_task],
            unscheduled_tasks=[],
            warnings=[],
        )
        slots = clickup_helper.find_free_slots(
            snap,
            cfg=cfg,
            duration_minutes=60,
            list_key='plan',
            start_after=datetime(2030, 3, 6, 0, 0, tzinfo=tz),
            max_slots=5,
        )
        self.assertGreaterEqual(len(slots), 1)
        self.assertEqual(slots[0].slot_start.date(), date(2030, 3, 7))

    def test_infer_task_policy(self):
        cfg = clickup_helper.HelperConfig()

        p_admin = clickup_helper.infer_task_policy('授業登録とTA申し込みを進める', 'auto', cfg)
        self.assertEqual(p_admin.category, 'admin')
        self.assertEqual(p_admin.priority_name, 'high')

        p_input = clickup_helper.infer_task_policy('inputファイルの整理', 'auto', cfg)
        self.assertEqual(p_input.category, 'input')
        self.assertEqual(p_input.priority_name, 'low')

        p_figure = clickup_helper.infer_task_policy('Figure 1の凡例を修正する', 'auto', cfg)
        self.assertEqual(p_figure.category, 'figure')
        self.assertEqual(p_figure.priority_name, 'normal')

        p_training = clickup_helper.infer_task_policy('training datasetを作る', 'auto', cfg)
        self.assertNotEqual(p_training.category, 'admin')

        p_ac = clickup_helper.infer_task_policy('AC', 'auto', cfg)
        self.assertEqual(p_ac.category, 'experiment')

        p_degas = clickup_helper.infer_task_policy('脱気', 'auto', cfg)
        self.assertEqual(p_degas.category, 'experiment')

        p_feedback = clickup_helper.infer_task_policy('指導教員からfeedbackをもらって修正する', 'auto', cfg)
        self.assertEqual(p_feedback.priority_name, 'urgent')

        p_input_marker = clickup_helper.infer_task_policy('input: paperを読む', 'auto', cfg)
        self.assertEqual(p_input_marker.category, 'input')

    def test_resolve_write_list_key_reroutes_blocked_plan(self):
        cfg = clickup_helper.HelperConfig(
            blocked_write_list_keys=['plan'],
            category_default_list={
                'admin': 'other',
                'writing': 'manuscript',
                'experiment': 'experiment',
                'figure': 'manuscript',
                'input': 'input',
                'general': 'other',
            },
        )
        resolved, warn = clickup_helper.resolve_write_list_key('plan', cfg, fallback_hint='other')
        self.assertEqual(resolved, 'other')
        self.assertIsNotNone(warn)
        self.assertIn('blocked', warn)

    def test_looks_fixed_keywords_for_meeting_gakkai_taikai(self):
        cfg = clickup_helper.HelperConfig(
            fixed_list_keys=['meeting', 'competition'],
            fixed_keyword_patterns=['meeting', '学会', '大会'],
        )
        for name in ['groupmeeting', '学会発表', '春季大会']:
            fixed, reason = clickup_helper.looks_fixed(
                {'name': name, 'description': '', 'status': {'type': 'open'}},
                list_key='other',
                cfg=cfg,
            )
            self.assertTrue(fixed)
            self.assertIn('matched keyword', reason)

    def test_select_best_slots_with_defer_bias(self):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        slots = [
            clickup_helper.SlotCandidate(
                label='direct',
                slot_start=now + timedelta(hours=1),
                slot_end=now + timedelta(hours=3),
                list_key='manuscript',
                score=0.0,
            ),
            clickup_helper.SlotCandidate(
                label='direct',
                slot_start=now + timedelta(hours=30),
                slot_end=now + timedelta(hours=32),
                list_key='manuscript',
                score=0.0,
            ),
        ]
        ranked = clickup_helper.select_best_slots(
            slots=slots,
            due_at=None,
            priority_name='low',
            schedule_bias_hours=24,
            max_candidates=2,
        )
        self.assertEqual(ranked[0].slot_start, slots[1].slot_start)

    def test_select_best_slots_prefers_tight_gap_when_enabled(self):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        early_wide = clickup_helper.SlotCandidate(
            label='direct',
            slot_start=now + timedelta(hours=1),
            slot_end=now + timedelta(hours=2),
            list_key='input',
            score=0.0,
            slack_minutes=600,
        )
        later_tight = clickup_helper.SlotCandidate(
            label='direct',
            slot_start=now + timedelta(hours=8),
            slot_end=now + timedelta(hours=9),
            list_key='input',
            score=0.0,
            slack_minutes=0,
        )
        normal_ranked = clickup_helper.select_best_slots(
            slots=[early_wide, later_tight],
            due_at=None,
            priority_name='low',
            schedule_bias_hours=0,
            max_candidates=2,
            prefer_tight_gap=False,
        )
        self.assertEqual(normal_ranked[0].slot_start, early_wide.slot_start)

        tight_ranked = clickup_helper.select_best_slots(
            slots=[early_wide, later_tight],
            due_at=None,
            priority_name='low',
            schedule_bias_hours=0,
            max_candidates=2,
            prefer_tight_gap=True,
        )
        self.assertEqual(tight_ranked[0].slot_start, later_tight.slot_start)

    def test_select_best_slots_prefers_jog_window_for_wait_like_task(self):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        early_slot = clickup_helper.SlotCandidate(
            label='direct',
            slot_start=now + timedelta(hours=6, minutes=30),
            slot_end=now + timedelta(hours=7, minutes=0),
            list_key='experiment',
            score=0.0,
        )
        near_jog_slot = clickup_helper.SlotCandidate(
            label='direct',
            slot_start=now + timedelta(hours=7, minutes=30),
            slot_end=now + timedelta(hours=8, minutes=0),
            list_key='experiment',
            score=0.0,
        )
        jog_intervals = [
            (now + timedelta(hours=8), now + timedelta(hours=9)),
        ]
        ranked = clickup_helper.select_best_slots(
            slots=[early_slot, near_jog_slot],
            due_at=None,
            priority_name='normal',
            schedule_bias_hours=0,
            max_candidates=2,
            prefer_around_jog_for_wait=True,
            jog_intervals=jog_intervals,
            jog_window_minutes=180,
        )
        self.assertEqual(ranked[0].slot_start, near_jog_slot.slot_start)

    def test_compute_wait_task_jog_bonus_prefers_gap_with_jog(self):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        jog_intervals = [
            (now + timedelta(hours=7), now + timedelta(hours=8)),
        ]
        gap_with_jog = clickup_helper.SlotCandidate(
            label='split',
            slot_start=now + timedelta(hours=6),
            slot_end=now + timedelta(hours=9, minutes=30),
            list_key='experiment',
            score=0.0,
            segments=[
                {
                    'slot_start': (now + timedelta(hours=6)).isoformat(),
                    'slot_end': (now + timedelta(hours=6, minutes=10)).isoformat(),
                },
                {
                    'slot_start': (now + timedelta(hours=9)).isoformat(),
                    'slot_end': (now + timedelta(hours=9, minutes=30)).isoformat(),
                },
            ],
        )
        gap_without_jog = clickup_helper.SlotCandidate(
            label='split',
            slot_start=now + timedelta(hours=1),
            slot_end=now + timedelta(hours=1, minutes=40),
            list_key='experiment',
            score=0.0,
            segments=[
                {
                    'slot_start': (now + timedelta(hours=1)).isoformat(),
                    'slot_end': (now + timedelta(hours=1, minutes=10)).isoformat(),
                },
                {
                    'slot_start': (now + timedelta(hours=1, minutes=30)).isoformat(),
                    'slot_end': (now + timedelta(hours=1, minutes=40)).isoformat(),
                },
            ],
        )
        with_bonus = clickup_helper.compute_wait_task_jog_bonus(
            gap_with_jog,
            jog_intervals=jog_intervals,
            window_minutes=180,
        )
        without_bonus = clickup_helper.compute_wait_task_jog_bonus(
            gap_without_jog,
            jog_intervals=jog_intervals,
            window_minutes=180,
        )
        self.assertGreater(with_bonus, without_bonus)

    def test_rearrangement_includes_target_list_and_prioritizes_input(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=13,
            min_gap_minutes=0,
            rearrange_allowed_list_keys=['input'],
        )
        t_input = clickup_helper.TaskWindow(
            task_id='ri',
            name='input task',
            list_id='li',
            list_key='input',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='none',
            start_at=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 6, 10, 0, tzinfo=tz),
            due_at=datetime(2030, 3, 6, 10, 0, tzinfo=tz),
            duration_minutes=60,
            movable=True,
        )
        t_manu = clickup_helper.TaskWindow(
            task_id='rm',
            name='manuscript task',
            list_id='lm',
            list_key='manuscript',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='none',
            start_at=datetime(2030, 3, 6, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 6, 11, 0, tzinfo=tz),
            due_at=datetime(2030, 3, 6, 11, 0, tzinfo=tz),
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[t_input, t_manu],
            scheduled_tasks=[t_input, t_manu],
            unscheduled_tasks=[],
            warnings=[],
        )
        candidates = clickup_helper.build_rearrangement_candidates(
            snapshot=snap,
            cfg=cfg,
            duration_minutes=60,
            list_key='manuscript',
            due_at=None,
            priority_name='high',
            schedule_bias_hours=0,
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_candidates=3,
        )
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0].moves[0]['task_id'], 'ri')
        self.assertTrue(any(c.moves[0]['task_id'] == 'rm' for c in candidates))

    def test_build_review_findings_detects_overlap(self):
        tz = ZoneInfo('Asia/Tokyo')
        today = datetime.now(tz).date()
        t1 = clickup_helper.TaskWindow(
            task_id='a1',
            name='A task',
            list_id='l1',
            list_key='plan',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=10),
            end_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=12),
            due_at=None,
            duration_minutes=120,
            movable=True,
        )
        t2 = clickup_helper.TaskWindow(
            task_id='a2',
            name='B task',
            list_id='l1',
            list_key='plan',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=11),
            end_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=13),
            due_at=None,
            duration_minutes=120,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=today,
            end_date=today + timedelta(days=2),
            timezone='Asia/Tokyo',
            fetched_tasks=[t1, t2],
            scheduled_tasks=[t1, t2],
            unscheduled_tasks=[],
            warnings=[],
        )
        cfg = clickup_helper.HelperConfig(work_hours_start=9, work_hours_end=22)
        findings = clickup_helper.build_review_findings(snap, cfg, near_days=2, max_findings=10)
        self.assertTrue(any(f['type'] == 'overlap' and f['task_id'] == 'a2' for f in findings))

    def test_build_review_findings_flags_low_priority_under_pressure(self):
        tz = ZoneInfo('Asia/Tokyo')
        now = datetime.now(tz).replace(second=0, microsecond=0)
        today = now.date()

        low_task = clickup_helper.TaskWindow(
            task_id='low1',
            name='inputメモ整理',
            list_id='l2',
            list_key='plan',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='none',
            start_at=now + timedelta(hours=1),
            end_at=now + timedelta(hours=3),
            due_at=None,
            duration_minutes=120,
            movable=True,
        )
        overdue_task = clickup_helper.TaskWindow(
            task_id='od1',
            name='締切タスク',
            list_id='l3',
            list_key='manuscript',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='high',
            start_at=None,
            end_at=None,
            due_at=now - timedelta(days=1),
            duration_minutes=120,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=today,
            end_date=today + timedelta(days=3),
            timezone='Asia/Tokyo',
            fetched_tasks=[low_task, overdue_task],
            scheduled_tasks=[low_task],
            unscheduled_tasks=[overdue_task],
            warnings=[],
        )
        cfg = clickup_helper.HelperConfig(work_hours_start=8, work_hours_end=22, deadline_warn_days=3)
        findings = clickup_helper.build_review_findings(snap, cfg, near_days=2, max_findings=10)
        self.assertTrue(any(f['type'] == 'defer_low_priority' and f['task_id'] == 'low1' for f in findings))

    def test_infer_workflow_tag_for_exp_code(self):
        cfg = clickup_helper.HelperConfig()
        tag_setup = clickup_helper.infer_workflow_tag('顕微鏡パソコンにclickup設定を入れる', 'code', cfg)
        self.assertIsNotNone(tag_setup)
        self.assertEqual(tag_setup.group, 'exp_code')
        self.assertEqual(tag_setup.stage, 'setup')

        tag_proc = clickup_helper.infer_workflow_tag('training datasetを作る', 'experiment', cfg)
        self.assertIsNotNone(tag_proc)
        self.assertEqual(tag_proc.group, 'exp_code')
        self.assertEqual(tag_proc.stage, 'processing')

    def test_infer_known_list_key(self):
        self.assertEqual(clickup_helper.infer_known_list_key('2_I N P U T ', 'Q P I'), 'input')
        self.assertEqual(clickup_helper.infer_known_list_key('6_S L I D E', 'Q P I'), 'slide')
        self.assertEqual(clickup_helper.infer_known_list_key('日常', 'O T H E R s'), 'daily')
        self.assertEqual(clickup_helper.infer_known_list_key('学会発表', 'M E E T I N G'), 'meeting')
        self.assertEqual(clickup_helper.infer_known_list_key('春季OP', 'T & F 大会'), 'competition')

    def test_build_workflow_order_findings(self):
        tz = ZoneInfo('Asia/Tokyo')
        today = datetime.now(tz).date()
        # processing が acquisition より先にある状態を作る
        t_proc = clickup_helper.TaskWindow(
            task_id='wf1',
            name='training datasetを作る',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=10),
            end_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=12),
            due_at=None,
            duration_minutes=120,
            movable=True,
        )
        t_acq = clickup_helper.TaskWindow(
            task_id='wf2',
            name='培地交換',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=14),
            end_at=datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(hour=15),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=today,
            end_date=today + timedelta(days=2),
            timezone='Asia/Tokyo',
            fetched_tasks=[t_proc, t_acq],
            scheduled_tasks=[t_proc, t_acq],
            unscheduled_tasks=[],
            warnings=[],
        )
        cfg = clickup_helper.HelperConfig(work_hours_start=9, work_hours_end=22, workflow_enabled=True)
        findings = clickup_helper.build_workflow_order_findings(snap, cfg, max_findings=5)
        self.assertTrue(any(f['type'] == 'workflow_order' and f['task_id'] == 'wf1' for f in findings))

    def test_medium_flow_step_and_start_after(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig()
        self.assertEqual(
            clickup_helper.infer_medium_flow_step('mNeonGreen計測、Sorbitol', 'experiment', cfg),
            'change_low_0',
        )
        self.assertEqual(
            clickup_helper.infer_medium_flow_step('【実験】2%培地交換（T0+48h）', 'experiment', cfg),
            'change_2_2',
        )
        self.assertEqual(
            clickup_helper.infer_medium_flow_step('【実験】0.0055%培地交換（T0+96h）', 'experiment', cfg),
            'change_2_low',
        )
        cfg_custom = clickup_helper.HelperConfig(medium_flow_t0_offset_map={48: 'change_0_2'})
        self.assertEqual(
            clickup_helper.infer_medium_flow_step('【実験】2%培地交換（T0+48h）', 'experiment', cfg_custom),
            'change_0_2',
        )

        t_cell = clickup_helper.TaskWindow(
            task_id='m1',
            name='細胞導入完了',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 1, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 1, 11, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        t_0_2 = clickup_helper.TaskWindow(
            task_id='m2',
            name='培地交換 0% 2%',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 8, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 8, 11, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 1),
            end_date=date(2030, 3, 15),
            timezone='Asia/Tokyo',
            fetched_tasks=[t_cell, t_0_2],
            scheduled_tasks=[t_cell, t_0_2],
            unscheduled_tasks=[],
            warnings=[],
        )
        start_after_early = clickup_helper.infer_medium_flow_start_after(
            snap,
            cfg,
            'change_2_2',
            reference_time=datetime(2030, 3, 5, 0, 0, tzinfo=tz),
        )
        self.assertEqual(start_after_early.date(), date(2030, 3, 3))

        start_after_late = clickup_helper.infer_medium_flow_start_after(
            snap,
            cfg,
            'change_2_2',
            reference_time=datetime(2030, 3, 10, 0, 0, tzinfo=tz),
        )
        self.assertEqual(start_after_late.date(), date(2030, 3, 10))

    def test_infer_medium_flow_end_before_on_culture(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(medium_flow_enabled=True)
        t_pre = clickup_helper.TaskWindow(
            task_id='pc1',
            name='pre culture',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 4, 12, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 4, 13, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 1),
            end_date=date(2030, 3, 7),
            timezone='Asia/Tokyo',
            fetched_tasks=[t_pre],
            scheduled_tasks=[t_pre],
            unscheduled_tasks=[],
            warnings=[],
        )
        end_before = clickup_helper.infer_medium_flow_end_before(
            snap,
            cfg,
            'on_culture',
            reference_time=datetime(2030, 3, 1, 0, 0, tzinfo=tz),
        )
        self.assertIsNotNone(end_before)
        self.assertEqual(end_before, datetime(2030, 3, 3, 12, 0, tzinfo=tz))

    def test_build_medium_flow_gap_findings(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(work_hours_start=9, work_hours_end=22, medium_flow_enabled=True)
        t_cell = clickup_helper.TaskWindow(
            task_id='mg1',
            name='細胞導入完了',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 1, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 1, 11, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        t_bad = clickup_helper.TaskWindow(
            task_id='mg2',
            name='培地交換 2% 2%',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 2, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 2, 11, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 1),
            end_date=date(2030, 3, 7),
            timezone='Asia/Tokyo',
            fetched_tasks=[t_cell, t_bad],
            scheduled_tasks=[t_cell, t_bad],
            unscheduled_tasks=[],
            warnings=[],
        )
        findings = clickup_helper.build_medium_flow_gap_findings(snap, cfg, max_findings=5)
        self.assertTrue(any(f['type'] == 'medium_flow_gap' and f['task_id'] == 'mg2' for f in findings))

    def test_build_medium_flow_gap_findings_detects_on_culture_too_late(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(work_hours_start=9, work_hours_end=22, medium_flow_enabled=True)
        t_on = clickup_helper.TaskWindow(
            task_id='mo1',
            name='O/N culture',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 6, 10, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 6, 11, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        t_pre = clickup_helper.TaskWindow(
            task_id='mo2',
            name='pre culture',
            list_id='l1',
            list_key='experiment',
            url='',
            status_name='to do',
            status_type='open',
            priority_name='normal',
            start_at=datetime(2030, 3, 6, 13, 0, tzinfo=tz),
            end_at=datetime(2030, 3, 6, 14, 0, tzinfo=tz),
            due_at=None,
            duration_minutes=60,
            movable=True,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 1),
            end_date=date(2030, 3, 7),
            timezone='Asia/Tokyo',
            fetched_tasks=[t_on, t_pre],
            scheduled_tasks=[t_on, t_pre],
            unscheduled_tasks=[],
            warnings=[],
        )
        findings = clickup_helper.build_medium_flow_gap_findings(snap, cfg, max_findings=10)
        self.assertTrue(any(f['type'] == 'medium_flow_before' and f['task_id'] == 'mo1' for f in findings))

    def test_infer_split_task_rule(self):
        cfg = clickup_helper.HelperConfig()
        rule = clickup_helper.infer_split_task_rule('glucose sln.作成', 'experiment', cfg)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.get('id'), 'glucose_prepare_collect')

    def test_infer_split_task_rule_nh4cl(self):
        cfg = clickup_helper.HelperConfig()
        rule = clickup_helper.infer_split_task_rule('NH4Cl作成', 'experiment', cfg)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.get('id'), 'nh4cl_prepare_collect')

    def test_infer_split_task_rule_emm(self):
        cfg = clickup_helper.HelperConfig()
        rule = clickup_helper.infer_split_task_rule('EMM培地作成', 'experiment', cfg)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.get('id'), 'emm_prepare_collect')

    def test_infer_split_task_rule_autoclave(self):
        cfg = clickup_helper.HelperConfig()
        for text in ['オートクレーブ', 'AC', 'チューブオークレ', 'autoclave']:
            rule = clickup_helper.infer_split_task_rule(text, 'experiment', cfg)
            self.assertIsNotNone(rule)
            self.assertEqual(rule.get('id'), 'autoclave_prepare_collect')

    def test_infer_split_task_rule_ac_boundary_safe(self):
        cfg = clickup_helper.HelperConfig()
        rule = clickup_helper.infer_split_task_rule('Macromolecular Crowdingの歴史を書く', 'experiment', cfg)
        self.assertIsNone(rule)

    def test_infer_split_task_rule_degas(self):
        cfg = clickup_helper.HelperConfig()
        for text in ['脱気', 'degas', 'degassing']:
            rule = clickup_helper.infer_split_task_rule(text, 'experiment', cfg)
            self.assertIsNotNone(rule)
            self.assertEqual(rule.get('id'), 'degas_prepare_collect')

    def test_find_split_slots_glucose_min_gap_same_day(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=22,
            min_gap_minutes=0,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[],
            scheduled_tasks=[],
            unscheduled_tasks=[],
            warnings=[],
        )
        rule = clickup_helper.infer_split_task_rule('glucose sln.作成', 'experiment', cfg)
        self.assertIsNotNone(rule)
        slots = clickup_helper.find_split_slots(
            snapshot=snap,
            cfg=cfg,
            list_key='experiment',
            base_name='glucose sln.作成',
            split_rule=rule,
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_slots=5,
        )
        self.assertGreaterEqual(len(slots), 1)
        first = slots[0]
        self.assertEqual(first.segments[0]['task_name'], 'glucose sln.作成')
        self.assertIn('回収', first.segments[1]['task_name'])
        prep_start = datetime.fromisoformat(first.segments[0]['slot_start'])
        collect_start = datetime.fromisoformat(first.segments[1]['slot_start'])
        self.assertGreaterEqual((collect_start - prep_start).total_seconds(), 6 * 3600)
        self.assertEqual(prep_start.date(), collect_start.date())

    def test_find_split_slots_nh4cl_min_gap_same_day(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=22,
            min_gap_minutes=0,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[],
            scheduled_tasks=[],
            unscheduled_tasks=[],
            warnings=[],
        )
        rule = clickup_helper.infer_split_task_rule('NH4Cl作成', 'experiment', cfg)
        self.assertIsNotNone(rule)
        slots = clickup_helper.find_split_slots(
            snapshot=snap,
            cfg=cfg,
            list_key='experiment',
            base_name='NH4Cl作成',
            split_rule=rule,
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_slots=5,
        )
        self.assertGreaterEqual(len(slots), 1)
        first = slots[0]
        self.assertEqual(first.segments[0]['task_name'], 'NH4Cl作成')
        self.assertIn('回収', first.segments[1]['task_name'])
        prep_start = datetime.fromisoformat(first.segments[0]['slot_start'])
        collect_start = datetime.fromisoformat(first.segments[1]['slot_start'])
        self.assertGreaterEqual((collect_start - prep_start).total_seconds(), 3600)
        self.assertEqual(prep_start.date(), collect_start.date())

    def test_find_split_slots_emm_min_gap_same_day(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=22,
            min_gap_minutes=0,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[],
            scheduled_tasks=[],
            unscheduled_tasks=[],
            warnings=[],
        )
        rule = clickup_helper.infer_split_task_rule('EMM培地作成', 'experiment', cfg)
        self.assertIsNotNone(rule)
        slots = clickup_helper.find_split_slots(
            snapshot=snap,
            cfg=cfg,
            list_key='experiment',
            base_name='EMM培地作成',
            split_rule=rule,
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_slots=5,
        )
        self.assertGreaterEqual(len(slots), 1)
        first = slots[0]
        self.assertEqual(first.segments[0]['task_name'], 'EMM培地作成')
        self.assertIn('回収', first.segments[1]['task_name'])
        prep_start = datetime.fromisoformat(first.segments[0]['slot_start'])
        collect_start = datetime.fromisoformat(first.segments[1]['slot_start'])
        prep_end = datetime.fromisoformat(first.segments[0]['slot_end'])
        collect_end = datetime.fromisoformat(first.segments[1]['slot_end'])
        self.assertEqual((prep_end - prep_start).total_seconds(), 30 * 60)
        self.assertEqual((collect_end - collect_start).total_seconds(), 10 * 60)
        self.assertGreaterEqual((collect_start - prep_start).total_seconds(), 2 * 3600)
        self.assertEqual(prep_start.date(), collect_start.date())

    def test_find_split_slots_autoclave_min_gap_same_day(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=22,
            min_gap_minutes=0,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[],
            scheduled_tasks=[],
            unscheduled_tasks=[],
            warnings=[],
        )
        rule = clickup_helper.infer_split_task_rule('オートクレーブ', 'experiment', cfg)
        self.assertIsNotNone(rule)
        slots = clickup_helper.find_split_slots(
            snapshot=snap,
            cfg=cfg,
            list_key='experiment',
            base_name='オートクレーブ',
            split_rule=rule,
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_slots=5,
        )
        self.assertGreaterEqual(len(slots), 1)
        first = slots[0]
        prep_start = datetime.fromisoformat(first.segments[0]['slot_start'])
        collect_start = datetime.fromisoformat(first.segments[1]['slot_start'])
        prep_end = datetime.fromisoformat(first.segments[0]['slot_end'])
        collect_end = datetime.fromisoformat(first.segments[1]['slot_end'])
        self.assertEqual((prep_end - prep_start).total_seconds(), 30 * 60)
        self.assertEqual((collect_end - collect_start).total_seconds(), 30 * 60)
        self.assertGreaterEqual((collect_start - prep_start).total_seconds(), 2 * 3600)
        self.assertEqual(prep_start.date(), collect_start.date())

    def test_find_split_slots_degas_min_gap_same_day(self):
        tz = ZoneInfo('Asia/Tokyo')
        cfg = clickup_helper.HelperConfig(
            work_hours_start=9,
            work_hours_end=22,
            min_gap_minutes=0,
        )
        snap = clickup_helper.AgendaSnapshot(
            start_date=date(2030, 3, 6),
            end_date=date(2030, 3, 6),
            timezone='Asia/Tokyo',
            fetched_tasks=[],
            scheduled_tasks=[],
            unscheduled_tasks=[],
            warnings=[],
        )
        rule = clickup_helper.infer_split_task_rule('脱気', 'experiment', cfg)
        self.assertIsNotNone(rule)
        slots = clickup_helper.find_split_slots(
            snapshot=snap,
            cfg=cfg,
            list_key='experiment',
            base_name='脱気',
            split_rule=rule,
            start_after=datetime(2030, 3, 6, 9, 0, tzinfo=tz),
            max_slots=5,
        )
        self.assertGreaterEqual(len(slots), 1)
        first = slots[0]
        prep_start = datetime.fromisoformat(first.segments[0]['slot_start'])
        collect_start = datetime.fromisoformat(first.segments[1]['slot_start'])
        prep_end = datetime.fromisoformat(first.segments[0]['slot_end'])
        collect_end = datetime.fromisoformat(first.segments[1]['slot_end'])
        self.assertEqual((prep_end - prep_start).total_seconds(), 15 * 60)
        self.assertEqual((collect_end - collect_start).total_seconds(), 15 * 60)
        self.assertGreaterEqual((collect_start - prep_start).total_seconds(), 3600)
        self.assertEqual(prep_start.date(), collect_start.date())


if __name__ == '__main__':
    unittest.main()
