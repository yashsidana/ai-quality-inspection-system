# 🚀 Streamlit Community Cloud 1-Click Deployment Guide

This guide walks you through deploying your **AI-Based Quality Inspection System** directly to **Streamlit Community Cloud** (`share.streamlit.io`) so that anyone (recruiters, managers, colleagues) can access your live application via a public URL.

---

## Prerequisites
1. Your repository pushed to GitHub at:
   `https://github.com/yashsidana/ai-quality-inspection-system`
   *(If not pushed yet, simply double-click `setup_git_and_push.bat` or run the git commands).*
2. A free account on [Streamlit Community Cloud](https://share.streamlit.io/) (sign in with your GitHub account).

---

## Step-by-Step Deployment (Takes < 2 Minutes)

### Step 1: Sign in to Streamlit Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io/)**.
2. Click **"Continue with GitHub"** to authorize Streamlit to access your GitHub repositories.

---

### Step 2: Create a New App
1. On your Streamlit Cloud workspace dashboard, click the **"Create app"** button (in the top right corner).
2. Choose **"Deploy a public app from GitHub"**.

---

### Step 3: Configure Repository Settings
Fill in the deployment form with the following details:

| Configuration Field | Value to Enter |
| :--- | :--- |
| **Repository** | `yashsidana/ai-quality-inspection-system` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **App URL (optional)** | `ai-quality-inspection.streamlit.app` *(or custom handle)* |

---

### Step 4 (Optional): Database & Cloud Secrets
Your application features **dual-mode database connectivity**:
* **Default Zero-Config**: If no secrets are set, it automatically creates and runs a local SQLite database (`quality_inspection.db`) with full functionality and sample data.
* **Production PostgreSQL**: If you want to connect to a cloud PostgreSQL instance (e.g., [Neon.tech](https://neon.tech), [Supabase](https://supabase.com), or [ElephantSQL](https://elephantsql.com)):
  1. In the deployment modal, click **"Advanced settings"**.
  2. In the **Secrets** box, paste:
     ```toml
     DATABASE_URL = "postgresql://your_user:your_password@your_host:5432/your_database"
     ```
  3. Click **Save**.

---

### Step 5: Click "Deploy!"
1. Click the **"Deploy!"** button.
2. Streamlit Cloud will automatically:
   - Provision a secure container.
   - Install system libraries listed in `packages.txt` (`libgl1-mesa-glx`, `libglib2.0-0`).
   - Install Python dependencies listed in `requirements.txt`.
   - Launch your interactive industrial inspection dashboard!
3. Within 60–90 seconds, your application will be live at:
   `https://ai-quality-inspection.streamlit.app`

---

## What Reviewers & Recruiters Will See:
- **🔬 Live Defect Inspection Station**: Interactive selector with sample manufacturing parts (PCBs, steel plates, gears, welds), defect bounding boxes, and severity rating.
- **📊 Production Analytics Dashboard**: First-pass yield KPI, Pareto defect breakdown, and filterable audit trail.
- **🎯 0.91 mAP Validation Benchmark**: Precision-recall curves, confusion matrices, and 40% inspection time reduction analysis.
- **⚡ FastAPI Interactive Simulator**: Live REST API testing demonstrating industrial microservice architecture.
