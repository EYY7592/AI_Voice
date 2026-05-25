import urllib.request
import json

# Create dummy wav file
dummy_wav_path = "dummy.wav"
with open(dummy_wav_path, "wb") as f:
    f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="audio"; filename="dummy.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode('utf-8') + open(dummy_wav_path, "rb").read() + f"\r\n--{boundary}\r\n".encode('utf-8') + (
    f'Content-Disposition: form-data; name="language"\r\n\r\n'
    f"zh\r\n"
    f"--{boundary}--\r\n"
).encode('utf-8')

req = urllib.request.Request(
    "http://localhost:7860/api/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode())
        print("Status:", response.status)
        print("Semantic Agent Result:")
        for agent in res.get("agents", []):
            if agent["name"] == "semantic":
                print(json.dumps(agent, indent=2, ensure_ascii=False))
except Exception as e:
    import traceback
    traceback.print_exc()
