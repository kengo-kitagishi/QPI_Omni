import os
import urllib.request, urllib.error, json

TOKEN = os.environ.get("NOTION_TOKEN")
if not TOKEN:
    raise SystemExit("Set NOTION_TOKEN (Notion integration secret).")
H = {"Authorization": "Bearer " + TOKEN, "Notion-Version": "2022-06-28", "Content-Type": "application/json"}

def api(method, path, data=None):
    url = "https://api.notion.com/v1" + path
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8")), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode("utf-8")), e.code
    except Exception as e:
        return {"error": str(e)}, 0

r, s = api("POST", "/search", {"filter": {"value": "database", "property": "object"}})
print("databases status:", s, "count:", len(r.get("results", [])))
for db in r.get("results", []):
    tp = db.get("title", [])
    title = "".join([x.get("plain_text","") for x in tp]) if tp else "(no title)"
    print("  DB:", db["id"], "|", title)
