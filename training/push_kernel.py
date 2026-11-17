import json, urllib.request

K = "KGAT_110d7edc73cc1e5ca37f7e17f68f5d6f"
nb = json.load(open("training/veqic_v1_kaggle.ipynb", encoding="utf-8"))

script = ["# Veqic 0.1 - QLoRA training on Gemma 4 E4B (Kaggle free GPU)\n",
          "# Data pulled live from the public GitHub repo - full transparency\n"]
for c in nb["cells"]:
    src = "".join(c["source"]).strip()
    if not src:
        continue
    if c["cell_type"] == "code":
        script.append(src + "\n")
    else:
        script.append("\n# --- " + src.split("\n")[0][:120] + " ---\n")

text = "".join(script)
payload = {
    "slug": "amirshraim1/veqic-0-1-training",
    "newTitle": "Veqic 0.1 Training (QLoRA on Gemma 4 E4B)",
    "text": text,
    "language": "python",
    "kernelType": "script",
    "isPrivate": True,
    "enableGpu": True,
}
req = urllib.request.Request(
    "https://www.kaggle.com/api/v1/kernels/push",
    data=json.dumps(payload).encode(),
    headers={"Authorization": "Bearer " + K, "Content-Type": "application/json"},
    method="POST",
)
try:
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print("version:", r.get("versionNumber"), "| error:", repr(r.get("error")), "| url:", r.get("url"))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:400])
