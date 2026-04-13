"""Extract all Japanese comments/strings from Python files."""
import re, os, json

jp_re = re.compile(r'[\u3041-\u3096\u30A1-\u30F6\u4E00-\u9FFF\u3000-\u303F]')

results = {}
for root, dirs, files in os.walk('scripts'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f).replace(os.sep, '/')
        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()
        file_jp = []
        for i, line in enumerate(lines, 1):
            if jp_re.search(line):
                file_jp.append((i, line.rstrip()))
        if file_jp:
            results[path] = file_jp

total = sum(len(v) for v in results.values())
print(f'Total: {total} Japanese lines across {len(results)} files')
with open('_jp_comments.json', 'w', encoding='utf-8') as f:
    json.dump({k: [(ln, txt) for ln, txt in v] for k, v in results.items()}, f, ensure_ascii=False, indent=2)
print('Saved to _jp_comments.json')
