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

## 2026-03-06 16:55 - apply by user approval (option 1)
- request: "alignment改善を 1時間後から2時間" with conflict-aware rearrangement
- applied: created `alignment改善` 2026-03-06 17:51-19:51 in `code` (task_id=86ewv38nt)
- moved: `学振DC1 執筆スケジュールを立てる` (86ewqyvyh) to 2026-03-06 21:40-22:40
- verification: same-day agenda rechecked, no overlap with `alignment改善` block

## 2026-03-06 18:15 - pending proposal (measurement-window use)
- request summary: during measurement windows, progress Claude Code based ClickUp scheduling setup and requirements definition
- constraint: analysis PC Claude Code integration is difficult for now; use waiting windows while measurements are running
- policy: confirm with user before applying any schedule change
- candidate windows found:
  - 2026-03-11 16:30-18:20 (between experiment blocks)
  - 2026-03-12 15:40-17:00 (between experiment blocks)
  - 2026-03-12 20:00-21:00 (post-measurement same day)

## 2026-03-06 18:18 - applied by user approval
- user approved proposal with "はい"
- created in `code` list:
  - 2026-03-11 16:30-18:20 `Claude CodeでClickUp連携の要件定義（計測中）` (86ewv3d5h)
  - 2026-03-12 15:40-17:00 `Claude CodeでClickUp連携の環境構築①（セットアップ準備）` (86ewv3d5p)
  - 2026-03-12 20:00-21:00 `Claude CodeでClickUp連携の環境構築②（検証手順・運用整理）` (86ewv3d62)
- verification: 2026-03-11~12 agenda checked, no overlaps, experiment blocks kept intact

## 2026-03-06 18:31 - applied by user approval
- user request: `2026/05/25 15:00~15:40 相関演習`
- created fixed event in `meeting`: `相関演習` (86ewv719w)
- time: 2026-05-25 15:00-15:40 JST
- verification: agenda(2026-05-25) shows 1 scheduled item, no conflict

## 2026-03-07 21:42 - applied by user approval (today carryover)
- context: user worked on grid_alignment + adjacent-diff code from 14:30 to 22:00, then dinner 1h; planned tasks delayed
- moved to 2026-03-08:
  - 10:40-11:40 `Macromolecular Crowdingの歴史を書く` (86evrwqv0)
  - 13:30-15:30 `学振DC1｜深掘り執筆セッションA-3/4（背景・独自性）` (86ewuzp7w)
  - 15:40-17:40 `執筆デー @ GRATBROWN Roast and Bake（駒場）` (86ewr34cy)
  - 18:50-19:20 `学振DC1 書き方の本を買って読む` (86ewqyvwr)
  - 19:30-20:30 `就活ES締め切り調査（花王・東京電力）` (86ewr35hx)
- verification: agenda/review for 2026-03-08 reports no overlap findings

## 2026-03-07 22:03 - done log applied
- user request: add today's work as done
- created and marked complete in `code`:
  - 14:30-18:00 `grid_alignment実装` (86ewvdw7r)
  - 18:00-22:00 `隣接差分コード実装` (86ewvdw8f)
- verification: both tasks show `status=complete` and `status_type=done`

## 2026-03-07 22:08 - applied by user request
- user request: move `細胞導入` to the morning of the day after tomorrow
- interpreted date: `2026-03-09` (明後日 from 2026-03-07 JST)
- moved: `細胞導入` (86ewr092c) -> 2026-03-09 08:30-10:00 JST
- verification: agenda(2026-03-09) shows the updated experiment slot in morning

## 2026-03-08 00:11 - applied by user request (medium exchange chain)
- user request: shift medium-exchange tasks together with moved cell introduction
- base anchor: `細胞導入` completion = 2026-03-09 10:00 JST
- recalculated by T0 offsets in task names:
  - `【実験】2%培地交換（T0+48h）` (86ewr0z00) -> 2026-03-11 10:00-11:00
  - `【実験】0.0055%培地交換（T0+96h）` (86ewr0z29) -> 2026-03-13 10:00-11:00
  - `【実験】0%培地交換（T0+120h）` (86ewr0z4d) -> 2026-03-14 10:00-11:00
  - `【実験】2%培地交換（T0+168h）` (86ewr0z63) -> 2026-03-16 10:00-11:00
  - `【実験】2%培地交換（T0+216h）` (86ewr0z90) -> 2026-03-18 10:00-11:00
- note: review found overlaps with non-experiment tasks after enforcing T0 timing; follow-up rescheduling can be applied if requested

## 2026-03-08 13:25 - applied by user request
- user report: worked on grid_aliment fixes from 11:00 to 13:30
- logged as done in `code`: `grid_alimentの修正` (86ewvgpn8), 2026-03-08 11:00-13:30, status=complete
- moved `細胞導入` (86ewr092c) to tomorrow morning: 2026-03-09 08:30-10:00
- re-anchored medium-exchange chain by T0 from new intro completion (10:00); resulting slots remained:
  - 03/11 10:00-11:00 (T0+48h), 03/13 10:00-11:00 (T0+96h), 03/14 10:00-11:00 (T0+120h), 03/16 10:00-11:00 (T0+168h), 03/18 10:00-11:00 (T0+216h)

## 2026-03-08 13:29 - applied by user approval (overlap cleanup)
- moved overlap tasks (non-experiment side) to clear conflicts:
  - `Macromolecular Crowdingの歴史を書く` (86evrwqv0) -> 2026-03-09 13:30-14:30
  - `学振DC1 書き方の本を買って読む` (86ewqyvwr) -> 2026-03-09 14:30-15:00
  - `ESどこに出せるか見る` (86ewuyegv) -> 2026-03-09 20:10-21:10
  - `Physical properties ...` (86evu49dq) -> 2026-03-12 13:40-15:40
- verification:
  - overlap findings were cleared in 2026-03-08~03-12 range
  - remaining review findings are workflow-order warnings only (no hard time overlap)

## 2026-03-08 13:31 - applied by user request
- user request: schedule execution of `41_.py` tonight
- created in `code`: `41_.py実行` (86ewvgq2x)
- time: 2026-03-08 19:30-20:30 JST
- verification: same-day agenda shows no overlap around the new slot

## 2026-03-08 13:33 - applied by user request
- user request: add `光学系の調整` tomorrow morning
- created in `experiment`: `光学系の調整` (86ewvgq5p)
- time: 2026-03-09 07:30-08:20 JST
- verification: agenda(2026-03-09) confirms no overlap with `細胞導入` (08:30-10:00)

## 2026-03-08 13:36 - policy update
- user rule: never overlap schedules with `部活` and `jog`
- config updated: `fixed_keyword_patterns` includes `部活`, `jog`, `ジョグ`
- effect: tasks matching these keywords are treated as fixed (`movable=False`) in advise/review/rearrange flows

## 2026-03-08 13:44 - policy update
- user preference: place wait-time tasks around jog to use gaps efficiently
- scheduling logic update:
  - detect wait-like tasks from split-task rules / wait keywords
  - collect jog anchors from scheduled tasks using `jog_keyword_patterns`
  - boost candidate ranking when wait tasks are adjacent to jog or when jog fits into split-task waiting gaps
- config keys added:
  - `jog_keyword_patterns`
  - `wait_task_use_jog_windows`
  - `wait_task_jog_window_minutes`

## 2026-03-08 13:46 - applied by user request
- user request: add `groupmeeting` on 2026-03-17 13:00-15:00
- created in `meeting`: `groupmeeting` (86ewvgray)
- verification: agenda(2026-03-17) confirms registration
- note: overlap detected with `学振DC1｜深掘り執筆セッションC（仮説と検証設計）` (12:20-14:20)

## 2026-03-08 13:52 - applied by user request
- user request: push writing task later to avoid groupmeeting overlap
- moved `学振DC1｜深掘り執筆セッションC（仮説と検証設計）` (86ewuzmqm)
  from 2026-03-17 12:20-14:20 -> 2026-03-17 15:00-17:00
- verification:
  - day review (2026-03-17): findings=0
  - 14-day review (2026-03-17..03-30): `after_due=0` (no deadline-late detected)

## 2026-03-08 13:59 - policy update
- user rule reinforced: never move/overlap `meeting`, `学会`, `大会`
- fixed-keyword guard expanded with `大会` in runtime config + default keywords
- validation: unit tests passed (`30` tests, includes fixed-keyword check for `groupmeeting` / `学会発表` / `春季大会`)

## 2026-03-08 14:08 - applied by user request
- user request: rename yesterday lunch task from generic `ご飯` to place name
- target identified (closed/cancelled): 2026-03-07 12:30-14:30 `ご飯` (86ewqrpah)
- renamed only task title -> `青龍門@渋谷` (status kept as `cancelled/closed`)

## 2026-03-08 14:22 - policy update
- user rule: `オートクレーブ / AC / チューブオークレ` is a split task
- added split rule `autoclave_prepare_collect`:
  - prepare: 30m
  - collect: 30m
  - constraint: collect >= 120m after prepare (same day)
- matching robustness:
  - short ASCII alias matching for split rules uses word-boundary style detection (prevents false positives like `Macromolecular` -> `ac`)
  - experiment category keywords extended with `autoclave`, `オートクレーブ`, `オークレ`, `ac`
- verification:
  - tests passed (`33` total)
  - `advise --text \"AC\"` now returns split candidates in `experiment` with 30m + 2h wait + 30m

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

## 2026-03-06 18:13 - advise consultation
- text: claude codeでclickup連携の環境構築をする。2時間。優先度中。計測中に進める
- category=general priority=normal bias=+0h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-181356.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-06 18:13 - advise consultation
- text: claude codeでclickup連携の要件定義をする。2時間。優先度高め
- category=general priority=normal bias=+0h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260306-181357.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-07 21:35 - review check
- range: 2026-03-06->2026-03-07
- pressure: overdue=87 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260307-213555.json

## 2026-03-07 21:36 - review check
- range: 2026-03-07->2026-03-08
- pressure: overdue=87 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260307-213619.json

## 2026-03-07 21:42 - review check
- range: 2026-03-08->2026-03-08
- pressure: overdue=83 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260307-214218.json

## 2026-03-08 00:11 - review check
- range: 2026-03-11->2026-03-18
- pressure: overdue=85 due_soon_unscheduled=0
- findings: 9 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-001146.json

## 2026-03-08 13:28 - review check
- range: 2026-03-08->2026-03-11
- pressure: overdue=85 due_soon_unscheduled=0
- findings: 5 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-132827.json

## 2026-03-08 13:29 - review check
- range: 2026-03-08->2026-03-12
- pressure: overdue=85 due_soon_unscheduled=0
- findings: 4 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-132922.json

## 2026-03-08 13:45 - advise consultation
- text: NH4Cl作成
- category=general priority=normal bias=+0h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=1 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-134529.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 13:46 - review check
- range: 2026-03-17->2026-03-17
- pressure: overdue=86 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-134658.json

## 2026-03-08 13:52 - review check
- range: 2026-03-17->2026-03-17
- pressure: overdue=86 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-135223.json

## 2026-03-08 13:52 - review check
- range: 2026-03-17->2026-03-17
- pressure: overdue=86 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-135254.json

## 2026-03-08 13:52 - review check
- range: 2026-03-17->2026-03-30
- pressure: overdue=86 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260308-135257.json

## 2026-03-08 14:22 - advise consultation
- text: AC
- category=experiment priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-142217.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 14:28 - advise consultation
- text: 脱気
- category=experiment priority=normal bias=+0h
- target_list=experiment due=(none) duration=30m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-142810.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 14:29 - policy update
- split-task rule added: `degas_prepare_collect`
- keywords: 脱気 / degas / degassing
- constraint: 15m prepare -> >=1h wait (same day) -> 15m collect
- validation: unit tests passed + advise live check ok

## 2026-03-08 14:36 - applied by user request
- user report: today's lunch was `マイバス寿司`
- renamed lunch task on 2026-03-08 13:30-14:00 in `plan`:
  - `ご飯` (86ewqrpb0) -> `マイバス寿司`
- left dinner task unchanged:
  - `ご飯` (86ewqrpbu) at 19:30-20:30

## 2026-03-08 20:41 - applied by user request
- user request: move `執筆デー` to tomorrow (2026-03-09)
- moved:
  - `執筆デー @ GRATBROWN Roast and Bake（駒場）` (86ewr34cy)
  - from 2026-03-08 21:45-23:45 -> 2026-03-09 12:10-14:10
- collision handling:
  - moved conflicting manuscript task
  - `学振DC1｜深掘り執筆セッションC-3/4（仮説と検証設計）` (86ewuzp8h)
  - from 2026-03-09 12:00-14:30 -> 2026-03-12 10:10-12:40
- verification:
  - agenda(2026-03-09): overlap-free with `執筆デー` in place
  - agenda(2026-03-12): moved manuscript task registered without overlap

## 2026-03-08 14:34 - advise consultation
- text: 脱気
- category=experiment priority=normal bias=+0h
- target_list=experiment due=(none) duration=30m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-143454.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 14:35 - advise consultation
- text: デバイス配合
- category=general priority=normal bias=+0h
- target_list=experiment due=(none) duration=30m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=1 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-143551.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 14:44 - advise consultation
- text: claude code for chromeでyoutubeの履歴を撮ってきてもらって分析してもらう
- category=general priority=low bias=+12h
- target_list=other due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-144409.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 14:45 - advise consultation
- text: リアルなOpticalsettingの図を作成する
- category=figure priority=normal bias=+8h
- target_list=manuscript due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-144557.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 20:40 - advise consultation
- text: 執筆デー @ GRATBROWN Roast and Bake（駒場）
- category=writing priority=high bias=-14h
- target_list=manuscript due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-204047.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 20:42 - advise consultation
- text: チューブオークレ
- category=experiment priority=normal bias=+0h
- target_list=experiment due=(none) duration=60m
- workflow=exp_code/unspecified/order999
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260308-204252.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-08 20:44 - applied by user request
- user request: add `兄の奥さんと会う` on night of 2026-03-15
- created in `daily`: `兄の奥さんと会う` (86eww81ra)
- time: 2026-03-15 19:00-21:00 JST
- verification: agenda(2026-03-15) confirms registration without overlap

## 2026-03-16 19:57 - review check
- range: 2026-03-15->2026-03-28
- pressure: overdue=122 due_soon_unscheduled=0
- findings: 3 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-195744.json

## 2026-03-16 19:58 - carryover planning
- from: 2026-03-15 targets=6 proposed=6
- committed: 0
- plan: /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260316-195858.json

## 2026-03-16 20:00 - carryover planning
- from: 2026-03-16 targets=6 proposed=5
- committed: 0
- plan: /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260316-200033.json

## 2026-03-16 20:03 - advise consultation
- text: 学振DC1｜深掘り執筆セッションC-2/4（仮説と検証設計）
- category=writing priority=high bias=-14h
- target_list=manuscript due=(none) duration=120m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260316-200307.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-16 20:03 - advise consultation
- text: 分裂様式について調べる。https://doi.org/10.1091/mbc.E14-10-1441
- category=general priority=normal bias=+0h
- target_list=manuscript due=(none) duration=180m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260316-200307.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-16 20:03 - advise consultation
- text: image processingの概要ず
- category=general priority=low bias=+12h
- target_list=slide due=(none) duration=60m
- workflow=none
- medium_flow=none
- candidates=3 plan=/Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-20260316-200308.json
- fixed_rules=meeting, competition
- rearrange_scope=input
- protect_experiment_structure=True

## 2026-03-16 20:05 - review check
- range: 2026-03-16->2026-03-29
- pressure: overdue=130 due_soon_unscheduled=0
- findings: 3 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-200507.json

## 2026-03-16 20:10 - daily backfill from obsidian
- sources:
  - /Users/kitak/Documents/Obsidian Vault/01_Daily/2026-03-09.md
  - /Users/kitak/Documents/Obsidian Vault/01_Daily/2026-03-10.md
  - /Users/kitak/Documents/Obsidian Vault/01_Daily/2026-03-11.md
  - /Users/kitak/Documents/Obsidian Vault/01_Daily/2026-03-12.md
  - /Users/kitak/Documents/Obsidian Vault/01_Daily/2026-03-13.md
  - /Users/kitak/Documents/Obsidian Vault/01_Daily/2026-03-14.md
- action: evidence-based past work added to ClickUp and marked complete
- added_done_tasks:
  - 86ewygpk5 qpi_fig_01_reconstruction_procedure Windows対応と単体パネル保存
  - 86ewygpke 逐次追跡モードとBFSグリッドキャリブレーション実装
  - 86ewygpkr 週次ログシステムSQLiteキャッシュ層実装
  - 86ewygpm1 W10週次レポート全面書き直し
  - 86ewygpmc リアルタイムドリフト補正タイムラプス立ち上げ・ホットフィックス
  - 86ewygpmm ECC外れ値フィルタ・MAD閾値調整・Pass3実装
  - 86ewygpmt QPI再構成とシフト計算の並列化
  - 86ewygpn1 日次ログ・週次レポート自動生成パイプライン構築
- verification: each task fetched again and confirmed status=complete

## 2026-03-16 20:18 - forward reschedule applied
- principle:
  - protect fixed events: meeting, competition, dinner, club
  - protect experiment structure before rescuing lower-priority input backlog
  - push low-priority input to 2026-03-23 instead of packing 2026-03-17..19
- moved_tasks:
  - 86ewr0z63 【実験】2%培地交換（T0+168h） -> 2026-03-17 09:00-10:00
  - 86ewr0z90 【実験】2%培地交換（T0+216h） -> 2026-03-19 09:00-10:00
  - 86ewr280b この2週間のログを読ませてどういうskill,rulesがいいかを考えてもらう。 -> 2026-03-19 10:10-13:10
  - 86ewuzp8t 学振DC1｜深掘り執筆セッションD-2/4（推敲・一貫性チェック） -> 2026-03-19 13:30-15:30
  - 86evw9qq1 分裂様式について調べる。https://doi.org/10.1091/mbc.E14-10-1441 -> 2026-03-23 09:00-12:00
  - 86evuek76 Evangelidis GD, Psarakis EZ... -> 2026-03-23 13:30-15:00
  - 86evu401t Nano Mid infra -> 2026-03-23 15:10-16:40
  - 86evu46zv Cell Volume change through water efflux impacts cell stiffness and stem cell fate -> 2026-03-23 16:50-17:50
- verification:
  - agenda checked for 2026-03-17..2026-03-26
  - review findings: 0
  - report: /Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-201842.json

## 2026-03-16 20:24 - overdue triage proposal
- current scan:
  - overdue tasks with past anchor and not done: 82
  - movable: 80
  - fixed: 2
- user-policy reaffirmed:
  - input and slide backlog should be deferred before squeezing writing or experiment-critical work
  - stale experiment chains must not be blindly reinserted; if the dependency chain is already broken, treat as redesign candidates
  - numbered writing workflow (A-3/4 -> A-4/4 etc.) should stay in order when possible
- proposed buckets:
  - rescue_this_month: recent manuscript/admin/figure work that supports DC1 and near-term deadlines
  - postpone_to_april: input, slide, documentation, low-priority other/code backlog
  - drop_or_redesign: December/January stale experiment chains, past-context daily tasks, outdated meeting leftovers

## 2026-03-16 22:20 - last-week carryover applied with cell-introduction priority
- user override:
  - fix `細胞導入` at 2026-03-17 morning as highest priority
  - move last week's unfinished work into this week or later
- manual reschedule highlights:
  - 86ewr092c 細胞導入 -> 2026-03-17 07:00-08:30
  - 86ewr0z00 / 86ewr0z29 / 86ewr0z4d / 86ewr0z63 / 86ewr0z90 -> re-anchored to the new T0 chain on 03-19 / 03-21 / 03-22 / 03-24 / 03-26
  - 学振 backlog A/B/C sessions inserted before or around existing A-4/B-4/C-4 blocks
  - admin/figure tasks moved to 03-19 and 03-27
- auto carryover applied:
  - 26 tasks from 2026-03-09..2026-03-15 moved into 2026-03-17..2026-04-01
  - low-priority input/slide/documentation deferred mainly to 2026-03-30..2026-04-01
- intentionally not moved:
  - 86ewuyzyk 福原と夜ご飯
  - 86eww81ra 兄の奥さんと会う
  - reason: past social events should not be blindly rescheduled
- validation:
  - custom club check after final adjustment: 0 conflicts
  - clickup review report: /Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-222043.json
  - note: review reported 4 workflow_order heuristics for experiment tasks, but no overlap findings; treated as workflow-tag overreach rather than timing conflict

## 2026-03-17 21:39 - competition days added
- added competition anchors:
  - 86ewyyzuc 学生個人選手権 -> 2026-04-25
  - 86ewyyzug 学生個人選手権 -> 2026-04-26
- rule applied:
  - competition days must stay otherwise empty
- conflict resolved:
  - 86ewv05bv Q-Microbio｜ポスター草案を教授に共有（レビュー①）
  - moved from 2026-04-25 10:00-11:00 to 2026-04-27 09:00-10:00
- verification:
  - 2026-04-25 and 2026-04-26 appear as blocked no-schedule days
  - review findings: 0
  - report: /Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260317-213954.json

## 2026-03-16 20:18 - review check
- range: 2026-03-17->2026-03-26
- pressure: overdue=127 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-201842.json

## 2026-03-16 20:22 - review check
- range: 2026-03-24->2026-03-31
- pressure: overdue=125 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-202220.json

## 2026-03-16 21:32 - review check
- range: 2026-03-17->2026-04-05
- pressure: overdue=103 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-213223.json

## 2026-03-16 22:20 - review check
- range: 2026-03-17->2026-04-05
- pressure: overdue=82 due_soon_unscheduled=0
- findings: 4 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260316-222043.json

## 2026-03-17 21:38 - review check
- range: 2026-04-23->2026-04-28
- pressure: overdue=89 due_soon_unscheduled=0
- findings: 1 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260317-213855.json

## 2026-03-17 21:39 - review check
- range: 2026-04-23->2026-04-28
- pressure: overdue=89 due_soon_unscheduled=0
- findings: 0 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260317-213954.json

## 2026-04-01 00:00 - hybrid scheduling policy adopted
- major policy change:
  - day-to-day schedule viewing and quick planning now defaults to Google Calendar
  - ClickUp remains the task/deadline/dependency system
- canonical note:
  - /Users/kitak/QPI_Omni/scripts/scheduling_hybrid_policy.md
- practical interpretation:
  - Google Calendar = visible daily calendar
  - ClickUp = structured task database

## 2026-04-06 13:36 - review check
- range: 2026-04-06->2026-05-30
- pressure: overdue=161 due_soon_unscheduled=0
- findings: 2 report=/Users/kitak/QPI_Omni/scripts/.clickup_reviews/review-20260406-133612.json

## 2026-04-06 13:37 - 基礎物理学実験TA fixed slots
- added fixed TA schedule blocks on 2026-04-06, 2026-04-20, 2026-04-27, 2026-05-07, 2026-05-11, 2026-05-18, 2026-05-25 at 12:30-17:00
- added `admin` list mapping to ClickUp helper config so agenda/review includes TA slots
- changed future `admin` category routing to write to the `ADMIN` ClickUp list by default
- marked `基礎物理学実験TA` as fixed via config keyword so auto-reschedule flows avoid moving it
