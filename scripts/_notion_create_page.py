"""Notion API script for creating/updating research note page"""
import os
import requests
import json

NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
if not NOTION_TOKEN:
    raise SystemExit("Set NOTION_TOKEN (Notion integration secret).")
DB_ID = "312eda96-228e-8165-9726-cd75b221357a"
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}
TODAY = "2026-02-26"
TITLE = "Notion\u9023\u643a\u30c4\u30fc\u30eb\u6574\u5099"


def search_today_pages():
    url = "https://api.notion.com/v1/search"
    payload = {
        "query": TODAY,
        "filter": {"value": "page", "property": "object"},
        "sort": {"direction": "descending", "timestamp": "last_edited_time"}
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    results = resp.json().get("results", [])
    found = []
    for page in results:
        if page.get("object") != "page":
            continue
        parent = page.get("parent", {})
        if parent.get("type") == "database_id":
            db_id = parent.get("database_id", "").replace("-", "")
            if DB_ID.replace("-", "") != db_id:
                continue
        title_prop = page.get("properties", {}).get("Name", {}) or page.get("properties", {}).get("title", {})
        title_list = title_prop.get("title", [])
        page_title = "".join([t.get("plain_text", "") for t in title_list])
        print(f"  Found page: '{page_title}' (id: {page['id']})")
        if TODAY in page_title and not page_title.startswith("[") :
            found.append(page)
    return found


def make_text_block(text, heading=None):
    rich_text = [{"type": "text", "text": {"content": text}}]
    if heading == 2:
        return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rich_text}}
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text}}


def make_bullet(text):
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def make_divider():
    return {"object": "block", "type": "divider", "divider": {}}


def build_page_blocks():
    return [
        make_text_block("\u4f55\u3092\u3057\u305f\u304b\u3063\u305f", heading=2),
        make_bullet("plan.md\u3092Notion\u306b\u4fdd\u5b58\u3067\u304d\u308b\u4ed5\u7d44\u307f\u3092\u6574\u5099\u3057\u305f\u3002\u307e\u305fNotion\u306e\u5404\u30da\u30fc\u30b8\u306bType\u30d7\u30ed\u30d1\u30c6\u30a3\u3067\u7a2e\u5225\uff08\u601d\u8003/\u4f5c\u696d\u30ed\u30b0/\u56f3/\u30d7\u30e9\u30f3\uff09\u3092\u81ea\u52d5\u8a2d\u5b9a\u3057\u305f\u304b\u3063\u305f\u3002"),
        make_text_block("\u4f55\u304c\u3067\u304d\u305f", heading=2),
        make_bullet("notion_plan_save.py \u306b\u91cd\u8907\u30c1\u30a7\u30c3\u30af\u3092\u8ffd\u52a0\u3002\u540c\u540d\u30da\u30fc\u30b8\u304c\u65e2\u5b58\u306e\u5834\u5408\u306f\u30b9\u30ad\u30c3\u30d7\u3057\u3066URL\u3092\u8868\u793a"),
        make_bullet("notion_setup_type.py \u3092\u65b0\u898f\u4f5c\u6210\u3002QPI Research Notes \u306b Type\uff08Select\uff09\u30d7\u30ed\u30d1\u30c6\u30a3\u3092\u8ffd\u52a0\uff08\u601d\u8003/\u4f5c\u696d\u30ed\u30b0/\u56f3/\u30d7\u30e9\u30f3\uff09"),
        make_bullet("notion_plan_save.py \u3067\u30d7\u30e9\u30f3\u4fdd\u5b58\u6642\u306b Type=\u30d7\u30e9\u30f3 \u3092\u81ea\u52d5\u8a2d\u5b9a"),
        make_bullet("figure_logger.py \u306e\u65b0\u898f\u30da\u30fc\u30b8\u4f5c\u6210\u6642\u306b Type=\u56f3 \u3092\u81ea\u52d5\u8a2d\u5b9a"),
        make_bullet("task-routing.mdc \u3092\u66f4\u65b0\u3002\u30d7\u30e9\u30f3\u4fdd\u5b58\u3092\u30a8\u30fc\u30b8\u30a7\u30f3\u30c8\u304c\u5b9f\u88c5\u5b8c\u4e86\u6642\u306b\u81ea\u52d5\u5b9f\u884c\u3059\u308b\u30eb\u30fc\u30eb\u306b\u5909\u66f4\uff08git commit\u306f\u4e0d\u8981\uff09"),
        make_text_block("\u4f5c\u6210\u30fb\u5909\u66f4\u3057\u305f\u30d5\u30a1\u30a4\u30eb", heading=2),
        make_bullet("scripts/notion_plan_save.py : _find_existing_plan_page() \u3092\u8ffd\u52a0\u3057\u91cd\u8907\u4fdd\u5b58\u3092\u9632\u6b62\u3002\u4fdd\u5b58\u6642\u306b Type=\u30d7\u30e9\u30f3 \u3092\u81ea\u52d5\u8a2d\u5b9a"),
        make_bullet("scripts/notion_setup_type.py : \u65b0\u898f\u4f5c\u6210\u3002DB\u306bType\u30d7\u30ed\u30d1\u30c6\u30a3\u3092\u8ffd\u52a0\u3059\u308b\u30e6\u30fc\u30c6\u30a3\u30ea\u30c6\u30a3"),
        make_bullet("scripts/figure_logger.py : \u65b0\u898f\u56f3\u30da\u30fc\u30b8\u4f5c\u6210\u6642\u306b Type=\u56f3 \u3092\u81ea\u52d5\u8a2d\u5b9a"),
        make_bullet(".cursor/rules/task-routing.mdc : \u30d7\u30e9\u30f3\u4fdd\u5b58\u30eb\u30fc\u30eb\u3092\u30a8\u30fc\u30b8\u30a7\u30f3\u30c8\u81ea\u52d5\u5b9f\u884c\u306b\u5909\u66f4\u3002\u4fdd\u5b58\u3057\u306a\u3044\u4f8b\u5916\u306b\u30eb\u30fc\u30eb\u8b70\u8ad6\u3092\u8ffd\u52a0"),
        make_text_block("\u8003\u3048\u305f\u3053\u3068\u30fb\u5224\u65ad\u3057\u305f\u3053\u3068", heading=2),
        make_bullet("\u65e5\u672c\u8a9e\u30d5\u30a1\u30a4\u30eb\u540d\u304cPowerShell\u3067\u6587\u5b57\u5316\u3051\u3059\u308b\u305f\u3081\u3001\u30cf\u30c3\u30b7\u30e5\u90e8\u5206\uff088\u6841\uff09\u3060\u3051\u3067\u691c\u7d22\u3059\u308b\u30d5\u30a9\u30fc\u30eb\u30d0\u30c3\u30af\u3092\u5b9f\u88c5\u3057\u305f"),
        make_bullet("notion_plan_save.py \u306e\u91cd\u8907\u30c1\u30a7\u30c3\u30af\u306fAPI\u306etitle\u30d5\u30a3\u30eb\u30bf\u3067\u5b9f\u73fe\uff08\u30ed\u30fc\u30ab\u30eb\u691c\u7d22\u3067\u306f\u306a\u304fNotion\u5074\u3067\u30d5\u30a3\u30eb\u30bf\uff09"),
        make_text_block("\u30d5\u30a3\u30fc\u30c9\u30d0\u30c3\u30af", heading=2),
        make_text_block("(\u7a7a\u6b04)"),
    ]


def create_page():
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"database_id": DB_ID},
        "properties": {
            "Name": {
                "title": [{"type": "text", "text": {"content": TITLE}}]
            },
            "Date": {
                "date": {"start": TODAY}
            },
            "Type": {"select": {"name": "\u4f5c\u696d\u30ed\u30b0"}},
            "Script": {"rich_text": [{"type": "text", "text": {"content": "scripts/notion_plan_save.py, scripts/notion_setup_type.py, scripts/figure_logger.py"}}]},
            "Description": {"rich_text": [{"type": "text", "text": {"content": "Notion\u9023\u643a\u30c4\u30fc\u30eb\u306e\u6574\u5099\uff08\u91cd\u8907\u30c1\u30a7\u30c3\u30af\u30fbType\u30d7\u30ed\u30d1\u30c6\u30a3\u81ea\u52d5\u8a2d\u5b9a\u30fb\u30d7\u30e9\u30f3\u81ea\u52d5\u4fdd\u5b58\uff09"}}]},
        },
        "children": build_page_blocks()
    }
    resp = requests.post(url, headers=HEADERS, json=payload)
    data = resp.json()
    if resp.status_code == 200:
        page_id = data["id"]
        page_url = data.get("url", f"https://www.notion.so/{page_id.replace('-', '')}")
        print(f"SUCCESS: page created")
        print(f"URL: {page_url}")
    else:
        print(f"ERROR: {resp.status_code}")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def append_to_page(page_id):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    blocks = [make_divider()] + build_page_blocks()
    payload = {"children": blocks}
    resp = requests.patch(url, headers=HEADERS, json=payload)
    data = resp.json()
    if resp.status_code == 200:
        print(f"SUCCESS: appended to page (id: {page_id})")
        print(f"URL: https://www.notion.so/{page_id.replace('-', '')}")
    else:
        print(f"ERROR: {resp.status_code}")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def search_related_pages():
    print("\n--- Searching related past pages ---")
    keywords = ["ClickUp", "task-routing", "rule"]
    all_found = {}
    for kw in keywords:
        url = "https://api.notion.com/v1/search"
        payload = {
            "query": kw,
            "filter": {"value": "page", "property": "object"},
            "sort": {"direction": "descending", "timestamp": "last_edited_time"}
        }
        resp = requests.post(url, headers=HEADERS, json=payload)
        for page in resp.json().get("results", []):
            pid = page.get("id")
            if pid in all_found:
                continue
            title_prop = page.get("properties", {}).get("Name", {}) or page.get("properties", {}).get("title", {})
            title_list = title_prop.get("title", [])
            page_title = "".join([t.get("plain_text", "") for t in title_list])
            if not page_title:
                continue
            if TITLE in page_title:
                continue
            page_url = page.get("url", f"https://www.notion.so/{pid.replace('-', '')}")
            date_prop = page.get("properties", {}).get("Date", {})
            date_val = date_prop.get("date", {}) if date_prop else {}
            date_str = date_val.get("start", "unknown") if date_val else "unknown"
            all_found[pid] = {"title": page_title, "url": page_url, "date": date_str, "keyword": kw}

    if all_found:
        print(f"Found {len(all_found)} related pages:")
        for pid, info in list(all_found.items())[:5]:
            print(f"  [{info['date']}] {info['title']}")
            print(f"  URL: {info['url']}")
    else:
        print("No related past pages found.")
    return all_found


if __name__ == "__main__":
    print(f"=== Searching today({TODAY}) work log pages ===")
    found_pages = search_today_pages()

    if found_pages:
        print(f"\nFound {len(found_pages)} existing page(s) -> will append")
        for p in found_pages:
            append_to_page(p["id"])
    else:
        print("\nNo existing page found -> creating new page")
        create_page()

    search_related_pages()
