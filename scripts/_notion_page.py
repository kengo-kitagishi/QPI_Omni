import os
import requests, json, sys
import urllib.request

TOKEN = os.environ.get("NOTION_TOKEN")
if not TOKEN:
    raise SystemExit("Set NOTION_TOKEN (Notion integration secret).")
DB_ID = "312eda96-228e-8165-9726-cd75b221357a"
H = {"Authorization": "Bearer " + TOKEN, "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
TODAY = "2026-02-25"
TITLE = "ClickUp\u9023\u643a\u306e\u4fee\u6b63\u3068\u30eb\u30fc\u30eb\u66f4\u65b0"

def rt(s): return [{"type":"text","text":{"content":s}}]
def h2(s): return {"object":"block","type":"heading_2","heading_2":{"rich_text":rt(s)}}
def p(s):  return {"object":"block","type":"paragraph","paragraph":{"rich_text":rt(s)}}
def bl(s): return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":rt(s)}}
def dv():  return {"object":"block","type":"divider","divider":{}}

BLOCKS = [
  h2("\u4f55\u3092\u3057\u305f\u304b\u3063\u305f"),
  bl("\u524d\u306e\u4f1a\u8a71\u3067\u300c\u3042\u3057\u305f\u300d+\u300c\u8abf\u3079\u307e\u3057\u3087\u3046\u300d\u3068\u8a00\u3063\u305f\u306e\u306bClickUp\u306b\u5165\u3089\u306a\u304b\u3063\u305f\u539f\u56e0\u3092\u8abf\u3079\u3066\u4fee\u6b63\u3057\u305f\u304b\u3063\u305f"),
  bl("\u300c\u8abf\u3079\u307e\u3057\u3087\u3046\u300d\u300c\u8abf\u3079\u3088\u3046\u300d\u306a\u3069\u306e\u767a\u8a00\u3092\u81ea\u52d5\u3067ClickUp\u30bf\u30b9\u30af\u306b\u767b\u9332\u3055\u308c\u308b\u3088\u3046\u306b\u3057\u305f\u304b\u3063\u305f"),
  h2("\u4f55\u304c\u3067\u304d\u305f"),
  bl("clickup_helper.py \u306e from pathlib import Path \u304c\u6292\u3051\u3066\u3044\u305f\u30d0\u30b0\u3092\u4fee\u6b63\u3057\u305f"),
  bl("task-routing.mdc \u306b\u300c\u8abf\u3079\u307e\u3057\u3087\u3046\u300d\u300c\u8abf\u3079\u3088\u3046\u300d\u300c\u8abf\u3079\u3066\u304a\u304d\u305f\u3044\u300d\u306a\u3069\u306eClickUp\u30c8\u30ea\u30ac\u30fc\u30ad\u30fc\u30ef\u30fc\u30c9\u3092\u8ffd\u52a0\u3057\u305f"),
  bl("\u660e\u65e5(2026-02-26)\u306e Claude Code / Cowork / Gemini CLI \u8abf\u67fb \u30bf\u30b9\u30af\u3092ClickUp\u306b\u767b\u9332\u3057\u305f"),
  h2("\u4f5c\u6210\u30fb\u5909\u66f4\u3057\u305f\u30d5\u30a1\u30a4\u30eb"),
  bl("scripts/clickup_helper.py : from pathlib import Path \u306e\u30a4\u30f3\u30dd\u30fc\u30c8\u304c\u6291\u3051\u3066\u3044\u305f\u305f\u3081\u5b9f\u884c\u6642\u30a8\u30e9\u30fc\u306b\u306a\u3063\u3066\u3044\u305f\u3002\u30a4\u30f3\u30dd\u30fc\u30c8\u3092\u8ffd\u52a0\u3057\u3066\u4fee\u6b63\u3002"),
  bl(".cursor/rules/task-routing.mdc : \u300c\u8abf\u3079\u307e\u3057\u3087\u3046\u300d\u300c\u8abf\u3079\u3088\u3046\u300d\u300c\u8abf\u3079\u3066\u304a\u304d\u305f\u3044\u300d\u300c\u30ea\u30b5\u30fc\u30c1\u3057\u305f\u3044\u300d\u306a\u3069\u3092ClickUp\u30c8\u30ea\u30ac\u30fc\u30ad\u30fc\u30ef\u30fc\u30c9\u3068\u3057\u3066\u8ffd\u52a0\u3057\u305f\u3002"),
  h2("\u8003\u3048\u305f\u3053\u3068\u30fb\u5224\u65ad\u3057\u305f\u3053\u3068"),
  bl("\u300c\u3042\u3057\u305f+\u8abf\u3079\u307e\u3057\u3087\u3046\u300d\u3068\u3044\u3046\u81ea\u7136\u306a\u767a\u8a00\u304cClickUp\u306b\u5165\u3089\u306a\u304b\u3063\u305f\u3002\u30eb\u30fc\u30eb\u306e\u6291\u3051\u3060\u3063\u305f\u3002"),
  bl("\u300c\u3057\u307e\u3057\u3087\u3046\u300d+\u65e5\u4ed8\u304c\u4f34\u3046\u5834\u5408\u3082\u8ffd\u52a0\u3059\u308b\u3053\u3068\u3067\u3001\u3088\u308a\u81ea\u7136\u306a\u767a\u8a00\u3067ClickUp\u306b\u767b\u9332\u3067\u304d\u308b\u3088\u3046\u306b\u3057\u305f\u3002"),
  h2("\u30d5\u30a3\u30fc\u30c9\u30d0\u30c3\u30af"),
  p("(\u7a7a\u6b04)"),
]

def api(method, path, data=None):
    url = "https://api.notion.com/v1" + path
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8")), r.status
    except Exception as e:
        return {"error": str(e)}, 0

print("Step1: Searching today pages...")
data, status = api("POST", "/search", {"query": TODAY, "filter": {"value":"page","property":"object"}})
found = []
if status == 200:
    for page in data.get("results", []):
        parent = page.get("parent", {})
        if parent.get("type") == "database_id" and parent.get("database_id","").replace("-","") == DB_ID.replace("-",""):
            tp = page.get("properties",{}).get("Name",{}) or page.get("properties",{}).get("title",{})
            title = "".join([x.get("plain_text","") for x in tp.get("title",[])])
            print(f"  Found: {title}")
            if TODAY in title and not title.startswith("["):
                found.append(page)
else:
    print(f"Search error: {data}")

if found:
    for pg in found:
        pid = pg["id"]
        r2, st2 = api("PATCH", f"/blocks/{pid}/children", {"children": [dv()] + BLOCKS})
        if st2 == 200:
            print(f"SUCCESS appended: https://notion.so/{pid.replace('-','')}")
        else:
            print(f"ERROR append {st2}: {r2}")
else:
    payload = {"parent":{"database_id":DB_ID},"properties":{"Name":{"title":[{"type":"text","text":{"content":TITLE}}]},"Date":{"date":{"start":TODAY}}},"children":BLOCKS}
    r2, st2 = api("POST", "/pages", payload)
    if st2 == 200:
        pid = r2["id"]
        print(f"SUCCESS created: https://notion.so/{pid.replace('-','')}")
    else:
        print(f"ERROR create {st2}: {r2}")

print("\nStep2: Searching related pages...")
found2 = {}
for kw in ["ClickUp", "task-routing"]:
    d2, _ = api("POST", "/search", {"query": kw, "filter":{"value":"page","property":"object"}})
    for pg in d2.get("results",[]):
        pid = pg["id"]
        if pid in found2: continue
        tp = pg.get("properties",{}).get("Name",{}) or pg.get("properties",{}).get("title",{})
        title = "".join([x.get("plain_text","") for x in tp.get("title",[])])
        if not title or TITLE in title: continue
        dp = pg.get("properties",{}).get("Date",{})
        dv2 = dp.get("date",{}) if dp else {}
        ds = dv2.get("start","unknown") if dv2 else "unknown"
        found2[pid] = {"title":title,"url":f"https://notion.so/{pid.replace('-','')}","date":ds}
if found2:
    print(f"Found {len(found2)} related page(s):")
    for pid, info in list(found2.items())[:5]:
        print(f"  [{info['date']}] {info['title']}")
        print(f"  -> {info['url']}")
else:
    print("No related pages found.")
