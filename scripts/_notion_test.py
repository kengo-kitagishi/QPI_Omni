import os
import urllib.request, urllib.error, json

TOKEN = os.environ.get("NOTION_TOKEN")
if not TOKEN:
    raise SystemExit("Set NOTION_TOKEN (Notion integration secret).")
DB_ID = "312eda96-228e-8165-9726-cd75b221357a"
H = {"Authorization": "Bearer " + TOKEN, "Notion-Version": "2022-06-28", "Content-Type": "application/json"}

def api(method, path, data=None):
    url = "https://api.notion.com/v1" + path
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8")), r.status
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        return json.loads(err_body) if err_body else {}, e.code
    except Exception as e:
        return {"error": str(e)}, 0

# Test: query the database directly
print("=== Testing database access ===")
r, s = api("POST", f"/databases/{DB_ID}/query", {"page_size": 3})
print(f"Status: {s}")
if s == 200:
    print(f"Results: {len(r.get('results',[]))}")
    for pg in r.get("results",[])[:3]:
        tp = pg.get("properties",{}).get("Name",{}) or pg.get("properties",{}).get("title",{})
        title = "".join([x.get("plain_text","") for x in tp.get("title",[])])
        print(f"  - {title}")
else:
    print(f"Error: {json.dumps(r, ensure_ascii=False)[:300]}")

# Test: try creating a simple page
print("\n=== Testing page creation ===")
payload = {
    "parent": {"database_id": DB_ID},
    "properties": {
        "Name": {"title": [{"type":"text","text":{"content":"TEST"}}]},
        "Date": {"date": {"start": "2026-02-25"}}
    }
}
r2, s2 = api("POST", "/pages", payload)
print(f"Status: {s2}")
if s2 == 200:
    print(f"Created page: {r2.get('id')}")
    # delete test page
    del_r, del_s = api("PATCH", f"/pages/{r2['id']}", {"archived": True})
    print(f"Cleaned up: {del_s}")
else:
    print(f"Error: {json.dumps(r2, ensure_ascii=False)[:500]}")
