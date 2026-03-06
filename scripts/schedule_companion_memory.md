# Schedule Companion Memory

## Core Rules
- `meeting` list and tasks matching `部活`, `授業`, `会議`, `発表` are treated as fixed and should not be moved automatically.
- `input`-like tasks are low priority and should not consume prime short-term focus slots.
- Figure editing tasks are medium-low; avoid pushing important deadlines by placing them too early.
- Writing/report/slides/manuscript tasks are high priority and should be secured first in 2-hour focus blocks.
- Experiment tasks are often re-planned, but ordering/structure should remain consistent when moved.

## Working Style
- Plan with monthly view + today's timeline + gantt.
- Re-check unfinished tasks from yesterday and re-plan them intentionally.
- Track conflicts, overdue tasks, and near-deadline unscheduled tasks before adding new work.
- Keep this file updated from each consultation to tune scheduling policy over time.

## Initial Setup Note (2026-03-06)
- Configured `clickup_helper.py` to auto-classify issue text into categories and propose schedules accordingly.
- Enabled dashboard and carryover workflows for month-scale planning and daily recovery.

## 2026-03-06 11:19 - dashboard check
- range: 2026-03-06->2026-04-09
- scheduled: 6 / unscheduled: 8
- overdue: 89 / due_soon_unscheduled: 0 / overlaps: 1
- dashboard: /Users/kitak/QPI_Omni/scripts/.clickup_dashboards/dashboard-20260306-111951.md

## 2026-03-06 11:20 - advise consultation
- text: この図を編集して報告資料を作る
- category=writing priority=high bias=-14h
- target_list=manuscript due=2026-03-10 duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112013.json
- fixed_rules=meeting
- rearrange_scope=experiment, manuscript, plan

## 2026-03-06 11:20 - advise consultation
- text: 授業の登録とTAの申し込みを今週中にやる
- category=admin priority=high bias=-16h
- target_list=plan due=2026-03-12 duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112014.json
- fixed_rules=meeting
- rearrange_scope=experiment, manuscript, plan

## 2026-03-06 11:21 - advise consultation
- text: 授業の登録とTAの申し込みを今週中にやる
- category=admin priority=high bias=-16h
- target_list=plan due=2026-03-12 duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112137.json
- fixed_rules=meeting
- rearrange_scope=experiment, manuscript, plan
- protect_experiment_structure=True

## 2026-03-06 11:21 - carryover planning
- from: 2026-03-05 targets=4 proposed=4
- committed: 0
- plan: /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112152.json

## 2026-03-06 11:22 - carryover planning
- from: 2026-03-05 targets=4 proposed=3
- committed: 0
- plan: /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112249.json

## 2026-03-06 11:24 - user preference update
- User plans with monthly calendar + today's calendar + gantt, and wants all three views for decision support.
- Main goal is focus protection: avoid low-priority figure/input work delaying deadline-critical tasks.
- Typical chat inputs are issue-like tasks (class registration, TA application, experiment validation, figure edit + report).
- Keep improving policy continuously through this memory file after each consultation.
- Default rearrangement policy switched to same-list only unless explicitly expanded.

## 2026-03-06 11:23 - advise consultation
- text: 授業の登録とTAの申し込みを今週中にやる
- category=admin priority=high bias=-16h
- target_list=plan due=2026-03-12 duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112355.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 11:25 - advise consultation
- text: TAのどの授業の担当になるか決めないとな
- category=admin priority=high bias=-16h
- target_list=plan due=(none) duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-112546.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 11:28 - response format preference
- Always return selection logic together with slot candidates.
- For each recommendation, include: category/priority inference, constraints used, and why higher-ranked than other slots.

## 2026-03-06 11:33 - review check
- range: 2026-03-06->2026-03-19
- pressure: overdue=85 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-113315.json

## 2026-03-06 11:33 - review check
- range: 2026-03-06->2026-03-19
- pressure: overdue=85 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-113356.json

## 2026-03-06 11:34 - advise consultation
- text: misumiのアルミフレームの購入をしないとなぁ
- category=general priority=normal bias=+0h
- target_list=plan due=(none) duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-113458.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 11:39 - review check
- range: 2026-03-06->2026-03-12
- pressure: overdue=85 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-113958.json

## 2026-03-06 11:41 - advise consultation
- text: 執筆デー @ GRATBROWN Roast and Bake（駒場）
- category=writing priority=high bias=-14h
- target_list=manuscript due=2026-03-10 duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-114118.json
- fixed_rules=meeting
- rearrange_scope=manuscript only
- protect_experiment_structure=True

## 2026-03-06 11:43 - weekly overdue replan consult
- Reviewed overdue tasks from 2026-02-27 to 2026-03-05.
- Total overdue in that week: 25 tasks.
- Kept experiment overdue tasks separate for explicit recovery decision instead of blind rescheduling.
- Proposed concrete rebook slots for top non-experiment overdue tasks with original-duration fallback.

## 2026-03-06 11:53 - experiment rebook from tomorrow
- Rebooked overdue experiment tasks to start from 2026-03-07.
- Updated 7 experiment tasks in sequence on 2026-03-07 to 2026-03-08 daytime.
- Kept existing upcoming experiment task on 2026-03-08 16:30 and 2026-03-10 15:00.
- Verification passed for all 7 task time updates.

## 2026-03-06 11:56 - experiment start from tomorrow
- User requested experiment tasks to start from tomorrow.
- Moved 7 overdue experiment tasks to 2026-03-07 and 2026-03-08 daytime.
- Also moved today's experiment task `86ewr0z4d` from 2026-03-06 15:00 to 2026-03-08 15:30.
- Verified schedule now has no experiment tasks on 2026-03-06 daytime.

## 2026-03-06 11:59 - advise consultation
- text: training datasetを作る
- category=admin priority=high bias=-16h
- target_list=experiment due=(none) duration=120m
- workflow=exp_code/processing/order30
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-115949.json
- fixed_rules=meeting
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:00 - advise consultation
- text: training datasetを作る
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=120m
- workflow=exp_code/processing/order30
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-120046.json
- fixed_rules=meeting
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:01 - review check
- range: 2026-03-06->2026-03-12
- pressure: overdue=69 due_soon_unscheduled=0
- findings: 4 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-120111.json

## 2026-03-06 12:01 - workflow-aware scheduling preference
- Treat `experiment` and `code` as one workflow family.
- Scheduling should consider content keywords and stage order, not only free slots.
- Added workflow-stage checks in advise/review and included reasoning in outputs.

## 2026-03-06 12:02 - advise consultation
- text: training datasetを作る
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=120m
- workflow=exp_code/processing/order30
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-120225.json
- fixed_rules=meeting
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:21 - dashboard check
- range: 2026-03-01->2026-03-31
- scheduled: 53 / unscheduled: 9
- overdue: 82 / due_soon_unscheduled: 0 / overlaps: 8
- dashboard: /Users/kitak/QPI_Omni/scripts/.clickup_dashboards/dashboard-20260306-122122.md

## 2026-03-06 12:21 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=82 due_soon_unscheduled=0
- findings: 16 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-122125.json

## 2026-03-06 12:23 - advise consultation
- text: 花王ES: 推敲して提出する
- category=general priority=high bias=-8h
- target_list=plan due=2026-03-12 duration=90m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-122300.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:23 - advise consultation
- text: 花王ES: 初稿を書く
- category=general priority=high bias=-8h
- target_list=plan due=2026-03-11 duration=120m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-122301.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:23 - advise consultation
- text: 花王ES: 設問確認と必要情報の洗い出し
- category=general priority=high bias=-8h
- target_list=plan due=2026-03-10 duration=60m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-122303.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:23 - advise consultation
- text: 花王ES: 推敲して提出する
- category=general priority=high bias=-8h
- target_list=plan due=2026-03-12 duration=90m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-122356.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:29 - advise consultation
- text: 先生のfeedbackを反映して修正
- category=general priority=urgent bias=-24h
- target_list=plan due=(none) duration=120m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-122919.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:29 - advise consultation
- text: input: Quantitative phase imaging paperを読む
- category=writing priority=high bias=-14h
- target_list=manuscript due=(none) duration=120m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-122919.json
- fixed_rules=meeting
- rearrange_scope=manuscript only
- protect_experiment_structure=True

## 2026-03-06 12:30 - advise consultation
- text: input: Quantitative phase imaging paperを読む
- category=writing priority=high bias=-14h
- target_list=manuscript due=(none) duration=120m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-123009.json
- fixed_rules=meeting
- rearrange_scope=manuscript only
- protect_experiment_structure=True

## 2026-03-06 12:31 - advise consultation
- text: input: Quantitative phase imaging paperを読む
- category=input priority=low bias=+36h
- target_list=input due=(none) duration=120m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-123110.json
- fixed_rules=meeting
- rearrange_scope=input only
- protect_experiment_structure=True

## 2026-03-06 12:37 - advise consultation
- text: 講習会受ける申し込みする
- category=admin priority=high bias=-16h
- target_list=plan due=2026-03-15 duration=60m
- workflow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-123729.json
- fixed_rules=meeting
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:42 - advise consultation
- text: 資料作成を1時間
- category=writing priority=normal bias=-6h
- target_list=plan due=(none) duration=60m
- workflow=none
- candidates=2 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-124221.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:42 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=82 due_soon_unscheduled=0
- findings: 16 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-124243.json

## 2026-03-06 12:44 - advise consultation
- text: ESどこに出せるかを調べる
- category=general priority=high bias=-8h
- target_list=plan due=2026-03-08 duration=60m
- workflow=none
- candidates=1 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-124427.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 12:52 - advise consultation
- text: 培地交換 2% 2%
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/acquisition/order20
- medium_flow=change_2_2
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-125238.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:52 - advise consultation
- text: mNeonGreen計測、Sorbitol
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/unspecified/order999
- medium_flow=change_low_0
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-125238.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:52 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=82 due_soon_unscheduled=0
- findings: 16 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-125257.json

## 2026-03-06 12:54 - advise consultation
- text: 培地交換 2% 2%
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/acquisition/order20
- medium_flow=change_2_2@2026-03-10T09:00:00+09:00
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-125408.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:54 - advise consultation
- text: mNeonGreen計測、Sorbitol
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/unspecified/order999
- medium_flow=change_low_0
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-125431.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 12:54 - advise consultation
- text: 培地交換2%Low%
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/acquisition/order20
- medium_flow=change_2_low
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-125455.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 13:02 - advise consultation
- text: glucose sln.作成
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=120m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-130217.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 13:02 - advise consultation
- text: glucose sln.作成
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=40m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-130252.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 13:12 - advise consultation
- text: NH4Cl作成
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-131205.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 13:12 - advise consultation
- text: O/N culture
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=120m
- workflow=exp_code/unspecified/order999
- medium_flow=on_culture
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-131209.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 13:14 - advise consultation
- text: ズボンを買いに行く
- category=general priority=normal bias=+0h
- target_list=daily due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=1 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-131422.json
- fixed_rules=meeting, competition
- rearrange_scope=daily only
- protect_experiment_structure=True

## 2026-03-06 13:17 - advise consultation
- text: EMM培地作成
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=40m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-131726.json
- fixed_rules=meeting, competition
- rearrange_scope=experiment only
- protect_experiment_structure=True

## 2026-03-06 13:20 - advise consultation
- text: 学会要旨・発表内容の最終確認
- category=general priority=normal bias=+0h
- target_list=plan due=2026-05-15 duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-132001.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 13:20 - advise consultation
- text: 移動・宿・当日導線の確認
- category=general priority=normal bias=+0h
- target_list=plan due=2026-05-14 duration=60m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-132003.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 13:20 - advise consultation
- text: 持ち物準備（PC・充電器・名刺）
- category=general priority=normal bias=+0h
- target_list=plan due=2026-05-17 duration=60m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-132003.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 13:20 - advise consultation
- text: 移動・宿・当日導線の確認
- category=general priority=normal bias=+0h
- target_list=plan due=2026-05-14 duration=60m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-132029.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 13:20 - advise consultation
- text: 持ち物準備（PC・充電器・名刺）
- category=general priority=normal bias=+0h
- target_list=plan due=2026-05-17 duration=60m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-132033.json
- fixed_rules=meeting, competition
- rearrange_scope=plan only
- protect_experiment_structure=True

## 2026-03-06 13:26 - figfix sync
- hub_root: /Users/kitak/Desktop/figure-hub
- mode: preview
- recommend_first: True
- list_key: (config default)
- created=5 skipped=0 failed=0

## 2026-03-06 13:27 - figfix sync
- hub_root: /Users/kitak/Desktop/figure-hub
- mode: preview
- recommend_first: True
- list_key: (config default)
- created=5 skipped=0 failed=0

## 2026-03-06 13:27 - review check
- range: 2026-03-06->2026-04-04
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 16 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-132759.json

## 2026-03-06 13:28 - carryover planning
- from: 2026-03-05 targets=0 proposed=0
- committed: 0
- plan: /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-132810.json

## 2026-03-06 13:31 - review check
- range: 2026-03-06->2026-04-04
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 9 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-133110.json

## 2026-03-06 13:32 - review check
- range: 2026-03-06->2026-04-04
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 7 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-133217.json

## 2026-03-06 13:36 - review check
- range: 2026-03-06->2026-04-04
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-133605.json

## 2026-03-06 13:42 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 7 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-134230.json

## 2026-03-06 13:43 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 7 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-134326.json

## 2026-03-06 13:46 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-134657.json

## 2026-03-06 13:53 - review check
- range: 2026-03-01->2026-03-31
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-135312.json

## 2026-03-06 13:54 - manual rebalance policy
- reason: next-week input concentration was too high
- policy: keep input load around <=240m/day, and prefer 1-2 blocks/day
- action: moved 5 input tasks from 2026-03-11/12 to 2026-03-16/17/18
- verification: next-week input load changed 03-11 360->120, 03-12 330->150, conflicts=0

## 2026-03-06 14:01 - medium exchange chain fix
- reason: medium-flow gap constraints were not applied to T0+xxh task names
- root cause: infer_medium_flow_step expected concentration-pair keywords; `T0+48h` style names were unmatched
- action: moved 5 experiment tasks to satisfy 2d->2d->1d->2d->2d gaps
- verification: chain order 48->96->120->168->216 now has gaps 2880/1510/2880/2880 minutes (required 2880/1440/2880/2880)
- hardening: added `medium_flow_t0_offset_map` and T0 offset inference in clickup_helper.py

## 2026-03-06 13:57 - figure-hub operation preference
- User preference: for any figure copy/move/adopt request, always follow figure-hub workflow instead of plain file copy as final operation.
- Required flow: `register -> use-sync -> status` as one logical sequence.
- Compatibility note: if CLI is still split, run `use -> sync` back-to-back as `use-sync` equivalent.
- If `project` / `project-root` is not specified, ask once before `use/sync` (except `thesis_overleaf` fixed root).
- Keep direct `cp/mv` only as temporary staging when explicitly needed, then normalize into figure-hub tracking.

## 2026-03-06 14:01 - review check
- range: 2026-03-07->2026-03-20
- pressure: overdue=79 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-140104.json

## 2026-03-06 14:01 - review check
- range: 2026-03-07->2026-03-20
- pressure: overdue=79 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-140154.json

## 2026-03-06 14:07 - input deferral default
- user preference: when adding a new task, existing input tasks can be moved later aggressively
- scheduler policy: include input in rearrangement scope by default (`rearrange_allowed_list_keys=["input"]`)
- ranking policy: input tasks are evaluated first as movable candidates in rearrangement
- compatibility: keep same-list rearrangement enabled to avoid regression

## 2026-03-06 14:27 - DC1 naming and priority policy
- user preference: prioritize `学振DC1` blocks for writing focus
- deadline rule clarification: "keep deadlines" means official deadlines after April, not old March DC1 draft tasks
- action: deleted 6 non-deadline DC1 tasks to free focus time
- operation rule: ask before applying future schedule edits; proposal-first workflow remains active

## 2026-03-06 14:31 - DC1 cleanup and Gakushin rebalance
- user preference update: remaining March `DC1` milestone/review tasks are unnecessary; remove them
- action: deleted 4 tasks (`第一稿提出`, `フィードバック反映①/②`, `第三者チェック依頼`)
- rebalance: moved 4 `学振DC1` deep sessions into the freed slots (03-07/03-08/03-09/03-22)
- verification: deep-writing total remains 32h (A/B/C/D each 8h), no newly created overlaps in future review window

## 2026-03-06 14:35 - Q-Microbio planning baseline
- user asked to create schedule for Q-Microbio including professor review, feedback fix, deadline, and travel expense filing
- event-day alignment: symposium updated to 2026-05-19 and 2026-05-20 (full-day blocks)
- created prep flow:
  - 04-29 初稿共有（教授レビュー依頼）
  - 05-02 フィードバック反映①
  - 05-08 旅費申請（提出）
  - 05-12 改訂版共有（最終確認）
  - 05-14 フィードバック反映②
  - 05-15 最終版確定・提出（自己締切）
  - 05-17 旅費申請ステータス確認
- note: official external deadline was not found in existing tasks; 05-15 is currently an internal deadline

## 2026-03-06 14:58 - Q-Microbio replan with official abstract deadline
- user clarified official abstract deadline is 2026-04-03 and symposium days are 2026-05-18/19
- action: rebuilt Q-Microbio timeline into two phases:
  - abstract submission phase (03-27 to 04-02)
  - conference preparation phase (04-18 to 05-17: poster, reviews, practice, printing)
- list policy applied: no new Q-Microbio tasks are placed in `plan` list; used `other` / `manuscript` only
- scheduler guard added: blocked write list key `plan` with auto-reroute

## 2026-03-06 14:06 - advise consultation
- text: TA担当を決める
- category=admin priority=high bias=-16h
- target_list=plan due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-140616.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-06 14:12 - review check
- range: 2026-03-10->2026-03-19
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-141251.json

## 2026-03-06 14:19 - review check
- range: 2026-03-11->2026-03-27
- pressure: overdue=79 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-141910.json

## 2026-03-06 14:19 - review check
- range: 2026-03-11->2026-03-27
- pressure: overdue=79 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-141950.json

## 2026-03-06 14:27 - review check
- range: 2026-03-06->2026-03-30
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-142735.json

## 2026-03-06 14:29 - review check
- range: 2026-03-06->2026-03-30
- pressure: overdue=77 due_soon_unscheduled=0
- findings: 4 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-142954.json

## 2026-03-06 14:30 - review check
- range: 2026-03-09->2026-03-30
- pressure: overdue=77 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-143054.json

## 2026-03-06 14:35 - review check
- range: 2026-04-25->2026-05-24
- pressure: overdue=77 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-143544.json

## 2026-03-06 14:57 - review check
- range: 2026-03-20->2026-06-07
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-145722.json

## 2026-03-06 14:57 - advise consultation
- text: 旅費申請をやる
- category=admin priority=high bias=-16h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-145728.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-06 14:58 - review check
- range: 2026-03-20->2026-06-07
- pressure: overdue=78 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260306-145810.json

## 2026-03-06 15:02 - advise consultation
- text: 授業登録をする
- category=admin priority=high bias=-16h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-150211.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-06 15:05 - policy update
- symposium dates fixed as user-confirmed: 2026-05-18 and 2026-05-19
- hard rule: do not place any future schedules into `plan` list (global, not only symposium)
- enforcement: keep `blocked_write_list_keys=["plan"]` and reroute writes to non-plan lists

## 2026-03-06 15:08 - advise consultation
- text: molecular crowding総説を読む。input。2時間。優先度低め
- category=input priority=low bias=+36h
- target_list=input due=(none) duration=120m
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-150834.json
- decision: keep as low-priority background reading unless manuscript section deadline is within 7 days

## 2026-03-06 15:07 - advise consultation
- text: molecular crowding review論文をinputする。2時間。優先度は低め
- category=input priority=urgent bias=+0h
- target_list=input due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-150750.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-06 15:08 - advise consultation
- text: molecular crowding総説を読む。input。2時間。優先度低め
- category=input priority=low bias=+36h
- target_list=input due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-150834.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-06 16:52 - advise consultation
- text: alignment改善をする。code。2時間。優先度高め
- category=general priority=normal bias=+0h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=1 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-165208.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True
