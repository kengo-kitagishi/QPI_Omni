"""Record this analysis occupancy and its immediate review task only."""
from datetime import datetime
from pathlib import Path
import json
from session_261017_recon import request


def main():
    record_path = Path('C:/QPI_analysis/261030/notion_tasks.json')
    record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {'pages': []}
    if len(record['pages']) == 2:
        return
    schema = request('GET', 'data_sources/7fb3f2db-e277-4883-b686-b364b2be9df7')
    date_key = next(k for k, v in schema['properties'].items() if v['type'] == 'date')
    record['date_key'] = date_key
    start = datetime.now().astimezone().isoformat(timespec='seconds')
    tasks = [
        ('【解析PC】261030 seg・tracking・系列HTML（whitelist除外、frame 430まで）', 'Remind'),
        ('【解析PC】261030 系列HTMLと欠測・分裂QCの確認', 'NextAction'),
    ]
    for title, status in tasks[len(record['pages']):]:
        props = {'Name': {'title': [{'text': {'content': title}}]},
                 'Status': {'select': {'name': status}}, 'Tag': {'multi_select': [{'name': 'QPI'}]}}
        if status == 'Remind':
            props[date_key] = {'date': {'start': start}}
        detail = ('対象578 ch、空のwhitelist 820 chを除外。seg frame 0–430、tracking/HTML frame 2–430。'
                  'D:/AquisitionData/Kitagishi/261030 の補正済み画像を C:/QPI_analysis/261030 に複製・SHA256照合。'
                  '同じC:にmask・CSV・HTMLを保存。全工程の所要時間・終了時刻は未実測。'
                  '測定後に占有終了を記録する。YEのRI校正は別途過去の培地差分撮影から確認中。')
        page = request('POST', 'pages', {'parent': {'type': 'data_source_id',
            'data_source_id': '7fb3f2db-e277-4883-b686-b364b2be9df7'}, 'properties': props,
            'children': [{'object': 'block', 'type': 'paragraph', 'paragraph': {'rich_text': [
                {'type': 'text', 'text': {'content': detail}}]}}]})
        record['pages'].append({'id': page['id'], 'url': page['url'], 'status': status})
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(status, page['url'], flush=True)


if __name__ == '__main__':
    main()
