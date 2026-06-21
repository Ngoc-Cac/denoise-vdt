import os

CACHE_DIR = ".cache"
ROOT_URL = "http://localhost:8080"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
