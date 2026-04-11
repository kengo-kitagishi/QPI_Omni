import os
import urllib.request, urllib.error, json

TOKEN = os.environ.get("NOTION_TOKEN")
if not TOKEN:
    raise SystemExit("Set NOTION_TOKEN (Notion integration secret).")
DB_ID = "312eda96-228e-8165-9726-cd75b221357a"
H = {"Authorization": "Bearer " + TOKEN, "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
TODAY = "2026-02-26"
TITLE = "scripts/archive/ \u306e\u6574\u5099"

def rt(s): return [{"type":"text","text":{"content":s}}]
def h2(s): return {"object":"block","type":"heading_2","heading_2":{"rich_text":rt(s)}}
def p(s):  return {"object":"block","type":"paragraph","paragraph":{"rich_text":rt(s)}}
def bl(s): return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":rt(s)}}
def dv():  return {"object":"block","type":"divider","divider":{}}

BLOCKS = [
  h2("\u4f55\u3092\u3057\u305f\u304b\u3063\u305f"),
  bl("\u4e0d\u8981\u306b\u306a\u3063\u305f\u30fb\u66f8\u304d\u76f4\u3057\u305f\u65e7\u30b9\u30af\u30ea\u30d7\u30c8\u3092\u6574\u7406\u3057\u305f\u304b\u3063\u305f"),
  bl("\u3069\u3053\u306b\u30a2\u30fc\u30ab\u30a4\u30d6\u3059\u308c\u3070\u3088\u3044\u304b\u308f\u304b\u3089\u306a\u304b\u3063\u305f\u306e\u3067\u7ba1\u7406\u65b9\u9488\u3092\u6c7a\u3081\u305f\u304b\u3063\u305f"),
  h2("\u4f55\u304c\u3067\u304d\u305f"),
  bl("scripts/archive/ \u30d5\u30a9\u30eb\u30c0\u3092\u65b0\u8a2d\u3057\u3001git mv \u3067\u30d5\u30a1\u30a4\u30eb\u3092\u79fb\u52d5\u3059\u308b\u904b\u7528\u30eb\u30fc\u30eb\u3092\u7b56\u5b9a\u3057\u305f"),
  bl("\u65e7\u30a2\u30e9\u30a4\u30e1\u30f3\u30c8\u7cfb\u30fb\u30bb\u30b0\u30e1\u30f3\u30c6\u30fc\u30b7\u30e7\u30f3\u7cfb\u30fbbacksub/diff\u7cfb\u306e\u8a08 11 \u30d5\u30a1\u30a4\u30eb\u3092\u30a2\u30fc\u30ab\u30a4\u30d6\u3057\u305f"),
  bl("git \u5c65\u6b74\u4e0a\u3067 rename \u3068\u3057\u3066\u8a18\u9332\u3055\u308c\u308b\u305f\u3081\u3001\u904e\u53bb\u306e\u30b3\u30fc\u30c9\u306f\u3044\u3064\u3067\u3082\u53c2\u7167\u53ef\u80fd\u306a\u72b6\u614b\u3092\u7dad\u6301"),
  h2("\u30a2\u30fc\u30ab\u30a4\u30d6\u3057\u305f\u30d5\u30a1\u30a4\u30eb\uff0811\u30d5\u30a1\u30a4\u30eb\uff09"),
  bl("03_alignment.py : ImageJ CSV\u30d9\u30fc\u30b9\u306e\u65e7\u30a2\u30e9\u30a4\u30e1\u30f3\u30c8\u624b\u6cd5"),
  bl("20_test_alignment_methods.py : \u30a2\u30e9\u30a4\u30e1\u30f3\u30c8\u624b\u6cd5\u306e\u6bd4\u8f03\u30c6\u30b9\u30c8\u7528"),
  bl("21_calc_alignment.py : ECC\u65b9\u5f0f\u306e\u4e2d\u9593\u30d0\u30fc\u30b8\u30e7\u30f3"),
  bl("22_ecc_alignment.py : ECC\u691c\u8a3c\u7528\u30d0\u30fc\u30b8\u30e7\u30f3"),
  bl("26-2-1_temp_batch.py : \u4e00\u6642\u30d0\u30c3\u30c1\u51e6\u7406\u30b9\u30af\u30ea\u30d7\u30c8"),
  bl("15_fluo_segment.py : CellposeOmni\u306b\u3088\u308b\u30bb\u30b0\u30e1\u30f3\u30c6\u30fc\u30b7\u30e7\u30f3\u8a66\u884c\u932f\u8aa4\u30b3\u30fc\u30c9\u7fa4"),
  bl("01_QPI_analysis.py : \u4f4d\u76f8\u518d\u69cb\u6210\u30d0\u30c3\u30c1\u51e6\u7406\u306e\u672a\u5b8c\u6210\u30b9\u30b1\u30eb\u30c8\u30f3"),
  bl("02_binary_outline.py : \u30de\u30b9\u30af\u304b\u3089\u8f2a\u90ed\u3092\u62bd\u51fa\u3059\u308b\u5f8c\u51e6\u7406\u30b9\u30af\u30ea\u30d7\u30c8"),
  bl("04_diff_from_first.py : 1\u679a\u76ee\u57fa\u6e96\u306e\u5dee\u5206\u753b\u50cf\u751f\u6210\uff08\u30cf\u30fc\u30c9\u30b3\u30fc\u30c9\u30d1\u30b9\u591a\u6570\uff09"),
  bl("05_2nd_backsub.py : \u30de\u30b9\u30af\u5e73\u5747\u5024\u306b\u3088\u308b\u80cc\u666f\u88dc\u6b63\uff08\u30cf\u30fc\u30c9\u30b3\u30fc\u30c9\u30d1\u30b9\u591a\u6570\uff09"),
  bl("14_medium_diff.py : medium\u5dee\u5206\u51e6\u7406\uff08\u65b0\u3057\u304f\u66f8\u304d\u76f4\u3059\u4e88\u5b9a\uff09"),
  h2("\u8003\u3048\u305f\u3053\u3068\u30fb\u5224\u65ad\u3057\u305f\u3053\u3068"),
  bl("git mv \u3092\u4f7f\u3046\u3053\u3068\u3067\u5c65\u6b74\u3092\u4fdd\u6301\u3057\u305f\u307e\u307e\u79fb\u52d5\u3067\u304d\u308b\u3002\u524a\u9664\u3067\u306f\u306a\u304f\u30a2\u30fc\u30ab\u30a4\u30d6\u306a\u306e\u3067\u904e\u53bb\u306e\u5b9f\u88c5\u3092\u53c2\u7167\u53ef\u80fd"),
  bl("_old_ \u3084 _deprecated_ \u30d7\u30ec\u30d5\u30a3\u30c3\u30af\u30b9\u306f\u4f7f\u308f\u305a archive/ \u306b\u96c6\u7d04\u3059\u308b\u65b9\u9488\u306b\u3057\u305f"),
  bl("\u30a2\u30fc\u30ab\u30a4\u30d6\u7406\u7531\u306f\u30b3\u30df\u30c3\u30c8\u30e1\u30c3\u30bb\u30fc\u30b8\u306b\u66f8\u304f\u3053\u3068\u3067\u3001\u306a\u305c\u305d\u306e\u624b\u6cd5\u3092\u3084\u3081\u305f\u304b\u306e\u8a18\u9332\u306b\u306a\u308b"),
  h2("\u30d5\u30a3\u30fc\u30c9\u30d0\u30c3\u30af"),
  p("(\u7a7a\u6b04)"),
]

def api(method, path, data=None):
    url = "https://api.notion.com/v1" + path
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8")), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode("utf-8")), e.code
    except Exception as e:
        return {"error": str(e)}, 0

# 莉頑律縺ｮ譌｢蟄倥・繝ｼ繧ｸ繧呈､懃ｴ｢
data, status = api("POST", "/search", {"query": TODAY, "filter": {"value":"page","property":"object"}})
found = []
if status == 200:
    for page in data.get("results", []):
        parent = page.get("parent", {})
        if parent.get("type") == "database_id" and parent.get("database_id","").replace("-","") == DB_ID.replace("-",""):
            tp = page.get("properties",{}).get("Name",{}) or page.get("properties",{}).get("title",{})
            title = "".join([x.get("plain_text","") for x in tp.get("title",[])])
            if TODAY in title and not title.startswith("["):
                found.append(page)
                print("found existing:", title)

if found:
    for pg in found:
        pid = pg["id"]
        r2, st2 = api("PATCH", "/blocks/" + pid + "/children", {"children": [dv()] + BLOCKS})
        if st2 == 200:
            print("appended: https://notion.so/" + pid.replace("-",""))
        else:
            print("ERROR append", st2, r2)
else:
    payload = {
        "parent": {"database_id": DB_ID},
        "properties": {
            "Name": {"title": [{"type":"text","text":{"content":TITLE}}]},
            "Date": {"date": {"start": TODAY}}
        },
        "children": BLOCKS
    }
    r2, st2 = api("POST", "/pages", payload)
    if st2 == 200:
        pid = r2["id"]
        print("created: https://notion.so/" + pid.replace("-",""))
    else:
        print("ERROR create", st2, json.dumps(r2, ensure_ascii=False)[:300])

# 髢｢騾｣繝壹・繧ｸ讀懃ｴ｢
print("\n--- related pages ---")
found2 = {}
for kw in ["archive", "\u30a2\u30e9\u30a4\u30e1\u30f3\u30c8", "\u30b9\u30af\u30ea\u30d7\u30c8"]:
    d2, _ = api("POST", "/search", {"query": kw, "filter":{"value":"page","property":"object"}})
    for pg in d2.get("results",[])[:5]:
        pid = pg["id"]
        if pid in found2: continue
        tp = pg.get("properties",{}).get("Name",{}) or pg.get("properties",{}).get("title",{})
        t = "".join([x.get("plain_text","") for x in tp.get("title",[])])
        if not t or TITLE in t: continue
        dp = pg.get("properties",{}).get("Date",{})
        ds = (dp.get("date",{}) or {}).get("start","?") if dp else "?"
        found2[pid] = {"title":t, "date":ds}
if found2:
    for pid, info in list(found2.items())[:5]:
        print("  [" + info["date"] + "] " + info["title"] + " -> https://notion.so/" + pid.replace("-",""))
else:
    print("  (none)")
