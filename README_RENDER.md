# Traffic Violation Backend — Render Deployment Guide

## Quick summary
This repo contains a FastAPI backend ready to deploy on Render using Docker.

## Files added
- Dockerfile
- requirements.txt (updated)
- app.py (FastAPI app)
- render.yaml (optional infra as code)

## How to deploy on Render (Docker)
1. Commit & push repository to GitHub (see commands below).
2. Go to https://dashboard.render.com → New → Web Service.
3. Select "Connect a repository" and choose this repository.
4. For Environment choose: **Docker**
   - Dockerfile Path: `Dockerfile`
5. Set Environment Variables (Render dashboard → Environment):
   - `API_USER` (optional) — recommended to set (e.g. admin)
   - `API_PASS` (optional) — recommended
   - `MODEL_PATH` (optional) — e.g. `/app/models/yolov8n.pt` or an S3 URL if you modify code to download it.
   - `MAX_UPLOAD_MB` (optional) — e.g. 50
6. Deploy.

## Local test (without Docker)
