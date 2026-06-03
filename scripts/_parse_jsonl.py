"""Temporary script to parse JSONL session logs."""
import json
import sys

def extract(path, label):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type", "")
            msg = obj.get("message", {})
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            role = msg.get("role", "")

            # User messages
            if t == "user" and role == "user":
                if isinstance(content, str):
                    text = content.strip()
                    if (text
                        and not text.startswith("<local-command")
                        and not text.startswith("<command-name")
                        and not text.startswith("<system-reminder")
                        and len(text) < 3000
                        and not text.startswith("<")):
                        print(f"\n[USER] {text[:800]}")
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            text = p["text"].strip()
                            if (text
                                and not text.startswith("<")
                                and len(text) < 3000):
                                print(f"\n[USER] {text[:800]}")

files = [
    (r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni\475ce7e8-c268-4b01-8f5b-a5b2079293ed.jsonl",
     "Session 2: 5/3 21:48 (ri-calibration-single-session-adapter)"),
    (r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni\3cfb1c11-2c1a-43fe-8348-8bb6a627f76a.jsonl",
     "Session 1: 5/3 21:57"),
    (r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni\e1a06914-67a1-48cf-99e4-85df82054471.jsonl",
     "Session 4: 5/4 12:44"),
    (r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni\2d7d07d8-972a-4a35-9591-269df658717c.jsonl",
     "Session 3: 5/4 14:08"),
    (r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni\bc8c146b-1931-45e5-a6c6-599a72e4f63b.jsonl",
     "Session 5: 5/4 17:53"),
]
for p, l in files:
    extract(p, l)
