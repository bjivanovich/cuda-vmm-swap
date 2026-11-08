import urllib.request, json, sys
payload = {"model":"architect","messages":[{"role":"system","content":"test"},{"role":"user","content":"hello"}]}
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions', data=data, headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('Status', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('Error', e)
