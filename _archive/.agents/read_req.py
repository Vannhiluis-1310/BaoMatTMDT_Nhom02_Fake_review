import json
with open('.agents/ORIGINAL_REQUEST.md', 'r', encoding='utf-8') as f:
    content = f.read()
with open('.agents/original_request.json', 'w', encoding='utf-8') as f:
    json.dump({'content': content}, f)
