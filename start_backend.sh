#!/bin/bash
cd /home/sprite/human-scene-interaction-system/backend
exec /.sprite/bin/python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
