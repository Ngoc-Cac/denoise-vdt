# Speech Denoising For Enhancing Downstream Meeting Transcription Service

## Prerequisite

- Python 3.14 is recommended.
- LFS must be initialised:
```bash
git lfs install
git lfs pull
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run local API server
```bash
python local_server.py

# In a different shell session
curl -X GET http://localhost:8080/health
# expected code 200 with output
# {
#     "status": "OK"
# }
```

3. (Optional) Run web UI with Gradio:
```bash
gradio gradio-app.py
```

**For running fine-tuning scripts**

Before running any of the notebooks or training scripts, pull the required
submodule:
```bash
# The MS-SNSD is very large so it will take some time.
git submodule update --init --recursive
```
