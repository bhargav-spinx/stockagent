# Auto-deploy (GitHub Actions → Google Cloud IAP SSH)

On every push to `main`, `.github/workflows/deploy.yml` authenticates to Google
Cloud, SSHes into the VM **through IAP** (no public IP, no port 22 open to the
internet), fast-forwards the code, and runs `deploy/deploy.sh` (installs deps if
`requirements.txt` changed, restarts `stockbot`, health-checks it).

This is a **one-time setup**. After it's done, deploys are automatic.

> All `gcloud` commands below can be run from the **Cloud Shell** in the GCP
> Console (no local `gcloud` needed). Replace `PROJECT_ID`, the zone, instance
> name (`stockbot`) and user (`ladanibhargav`) if yours differ.

---

## 1. Create a deploy service account

```bash
gcloud iam service-accounts create stockbot-deployer \
  --display-name="GitHub Actions deployer"

SA="stockbot-deployer@PROJECT_ID.iam.gserviceaccount.com"
```

## 2. Grant it the minimal roles

```bash
for ROLE in \
  roles/iap.tunnelResourceAccessor \
  roles/compute.instanceAdmin.v1 \
  roles/iam.serviceAccountUser ; do
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:$SA" --role="$ROLE"
done
```

- `iap.tunnelResourceAccessor` — open the IAP SSH tunnel
- `compute.instanceAdmin.v1` — let `gcloud compute ssh` push its ephemeral key to instance metadata
- `iam.serviceAccountUser` — act as the VM's service account

## 3. Create a JSON key for the service account

```bash
gcloud iam service-accounts keys create sa-key.json --iam-account="$SA"
```

`sa-key.json` is the value for the `GCP_SA_KEY` secret below. **Delete it locally
after pasting it into GitHub.**

## 4. Allow IAP to reach SSH (firewall — internal range only, NOT the internet)

```bash
gcloud compute firewall-rules create allow-iap-ssh \
  --direction=INGRESS --action=ALLOW --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 --network=default
```

`35.235.240.0/20` is Google's IAP range — the only source that can SSH. The VM
does **not** need a public IP.

## 5. Allow passwordless restart on the VM

The login user must restart the service without a password prompt:

```bash
# on the VM (Cloud Console SSH button), as ladanibhargav
echo 'ladanibhargav ALL=(root) NOPASSWD: /bin/systemctl restart stockbot, /bin/systemctl status stockbot' \
  | sudo tee /etc/sudoers.d/stockbot-deploy
sudo chmod 440 /etc/sudoers.d/stockbot-deploy
```

(Adjust username / `which systemctl` path if different.)

## 6. Add the repo secrets (GitHub → Settings → Secrets and variables → Actions)

| Secret | Value |
|---|---|
| `GCP_SA_KEY` | full contents of `sa-key.json` |
| `VM_INSTANCE` | `stockbot` |
| `VM_ZONE` | `us-central1-a` |
| `VM_USER` | `ladanibhargav` |
| `GCP_PROJECT` | your project id (optional — inferred from the key if omitted) |

## 7. Test

- Run it manually first: **Actions → Deploy to VM → Run workflow**.
- The final log lines should read `OK — stockbot active at <sha>`.
- Then any push to `main` deploys automatically.

---

## Notes & safety

- **More secure than open SSH:** no public IP, no port 22 to the internet —
  only IAP (`35.235.240.0/20`) reaches the VM, gated by IAM.
- **Secrets that stay on the VM** are untouched by deploy: `.env`
  (`MARKETAUX_API_TOKEN`, `ANGEL_*`, `TELETHON_*`, `TELEGRAM_BOT_TOKEN`) and
  `stockagent.db`. They're gitignored, and `git reset --hard` does not remove
  untracked/ignored files — keys and data survive every deploy.
- **First deploy of a new key** (e.g. `MARKETAUX_API_TOKEN`): the code arrives
  via deploy, but you must still add the key to `~/stockagent/.env` on the VM by
  hand — deploy will not create it.
- **Immediate restart:** a deploy during market hours (09:15–15:30 IST) briefly
  pauses scanning while the service restarts. To defer daytime restarts to
  after-close, gate `deploy.sh`'s restart on the clock — ask and I'll add it.
- **Rollback:** SSH in and `git reset --hard <old-sha> && bash deploy/deploy.sh`.
- **Even tighter (optional):** replace the JSON key with Workload Identity
  Federation (no long-lived key in GitHub at all) — more setup, ask if you want it.
