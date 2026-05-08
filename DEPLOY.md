# Deploy to Google Cloud Run

24/7 Telegram bot on Cloud Run, webhook mode. Free tier covers casual usage.

## One-time setup

1. **Install gcloud CLI** — https://cloud.google.com/sdk/docs/install (Windows: download the installer, run it, restart PowerShell).

2. **Create / pick a GCP project** and enable billing (Cloud Run requires billing on, but free tier still applies):
   ```powershell
   gcloud auth login
   gcloud projects create stockagent-bot --name="StockAgent"   # or pick an existing one
   gcloud config set project stockagent-bot
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
   ```

3. **Pick a region** — `asia-south1` (Mumbai) is closest to NSE / your users:
   ```powershell
   gcloud config set run/region asia-south1
   ```

## Deploy

From the repo root:

```powershell
gcloud run deploy stockagent `
    --source . `
    --platform managed `
    --allow-unauthenticated `
    --memory 512Mi `
    --cpu 1 `
    --timeout 300 `
    --set-env-vars "TELEGRAM_BOT_TOKEN=<your-token>,ANGEL_API_KEY=<key>,ANGEL_CLIENT_CODE=<code>,ANGEL_PASSWORD=<mpin>,ANGEL_TOTP_SECRET=<seed>,WEBHOOK_SECRET=<random-string>"
```

`gcloud` builds the image from the `Dockerfile`, pushes to Artifact Registry, and deploys. First deploy takes ~3-5 min. The command prints a service URL like:

```
https://stockagent-xxxxx-as.a.run.app
```

## Wire up the webhook

The bot needs `WEBHOOK_URL` to know its own public URL. Set it and redeploy:

```powershell
gcloud run services update stockagent `
    --update-env-vars "WEBHOOK_URL=https://stockagent-xxxxx-as.a.run.app"
```

When the container starts, `app.run_webhook(...)` automatically calls Telegram's `setWebhook` and points it at your URL. Test by sending `/start` to the bot in Telegram.

## Verify it works

```powershell
# Confirm Telegram knows about the webhook
$t = "<your-bot-token>"
Invoke-RestMethod "https://api.telegram.org/bot$t/getWebhookInfo"

# Tail Cloud Run logs
gcloud run services logs tail stockagent
```

## Production notes

- **In-memory watchlists are lost on cold start.** Each `/watch` is held in a Python dict; when Cloud Run scales to zero, the dict resets. Move to Firestore if you need watchlists to persist (small change in `bot.py` — let me know).
- **Don't enable `DISABLE_SSL_VERIFY` or `ANGEL_DISABLE_SSL` on Cloud Run.** Those are local-dev workarounds for your corp proxy. Cloud Run hits the public internet directly, so cert verification works normally.
- **Cold starts are 1-3 sec.** Telegram tolerates this. To eliminate them: `--min-instances=1` (~$5–15/mo) on the deploy command.
- **Secrets in env vars are visible in the Cloud Console.** For real prod, use Google Secret Manager instead of `--set-env-vars` (let me know if you want that wired in).

## Update / redeploy

After code changes:

```powershell
gcloud run deploy stockagent --source .
```

Env vars are preserved unless you pass `--set-env-vars` again.

## Tear down

```powershell
gcloud run services delete stockagent
```
