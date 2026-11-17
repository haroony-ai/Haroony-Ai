# Setting Up an Organization & Custom Domain for Haroony AI

Haroony AI currently runs automatically on GitHub Pages (`https://toolautomation2-droid.github.io/Haroony-Ai`).
Follow these steps to move it to a clean **Organization** and **Custom Domain** (for example, `https://haronyphysics.org` or `https://ai.haronyphysics.org`):

## 1. Create a Free GitHub Organization
1. On GitHub, click **+** (top-right) -> **New organization** -> choose **Free** ($0/month).
2. Name it something clean for your academy, e.g. `harony-physics` or `harony-clinic` (gives `https://github.com/harony-physics`).
3. Transfer this repository into that Organization (`Repository Settings` -> `General` -> `Danger Zone` -> `Transfer ownership`).
   - Once transferred, GitHub Pages automatically moves to `https://harony-physics.github.io/Haroony-Ai` (or `https://harony-physics.github.io` if the repo is named `harony-physics.github.io`).

## 2. Register a Good Organization Domain
Choose a short, trustworthy domain on Cloudflare Registrar, Namecheap, or Porkbun (typically ~$10–$12/year):
- `haronyphysics.org` (best for an educational clinic/organization)
- `haroonyai.org` or `haronyclinic.org`
- Or a subdomain like `ai.haronyphysics.com` if you already own a `.com` domain.

## 3. Connect the Domain to GitHub Pages (Free HTTPS Included)
### Option A — Apex Domain (e.g. `haronyphysics.org`)
Add these **4 `A` records** in your domain registrar's DNS panel (pointing `@` to GitHub Pages):
- `185.199.108.153`
- `185.199.109.153`
- `185.199.110.153`
- `185.199.111.153`

And add a **`CNAME`** record for `www`:
- `www` -> `toolautomation2-droid.github.io` (or `<your-org>.github.io`)

### Option B — Subdomain (e.g. `ai.haronyphysics.org`)
Add one **`CNAME`** record:
- `ai` -> `toolautomation2-droid.github.io` (or `<your-org>.github.io`)

## 4. Enable Custom Domain & HTTPS in GitHub
1. Open the repository on GitHub -> **Settings** -> **Pages**.
2. Under **Custom domain**, enter your domain (e.g. `haronyphysics.org`) and click **Save** (GitHub automatically commits the `CNAME` file once DNS resolves).
3. Wait a few minutes for DNS verification, then check **Enforce HTTPS**.
4. *(Optional)* If you also host the FastAPI backend (`uvicorn backend.app:app`) on Render / Railway / Fly.io / VPS with `.env` keys, point `api.haronyphysics.org` or your main domain to that server for full server-side multi-key LLM + RapidOCR support.
