# Auto-deploy (GitHub Actions → SSH)

On every push to `main`, `.github/workflows/deploy.yml` SSHes into the VM,
fast-forwards the code, and runs `deploy/deploy.sh` (installs deps if
`requirements.txt` changed, restarts `stockbot`, health-checks it).

This is a **one-time setup**. After it's done, deploys are automatic.

---

## 1. Generate a dedicated deploy SSH key (on your laptop)

```bash
ssh-keygen -t ed25519 -f stockbot_deploy -C "github-actions-deploy" -N ""
```

Produces `stockbot_deploy` (private) and `stockbot_deploy.pub` (public).

## 2. Authorize the key on the VM

Append the **public** key to the deploy user's authorized_keys (browser SSH via
the GCP Console works):

```bash
# on the VM, as the deploy user (e.g. ladanibhargav)
echo 'ssh-ed25519 AAAA...github-actions-deploy' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## 3. Allow passwordless restart on the VM

The deploy user must restart the service without a password prompt:

```bash
# on the VM
echo 'ladanibhargav ALL=(root) NOPASSWD: /bin/systemctl restart stockbot, /bin/systemctl status stockbot' \
  | sudo tee /etc/sudoers.d/stockbot-deploy
sudo chmod 440 /etc/sudoers.d/stockbot-deploy
```

(Adjust the username and the `systemctl` path — `which systemctl` — if different.)

## 4. Open SSH to GitHub's runners (GCP firewall)

The VM needs a **public IP** and inbound TCP **22** reachable from GitHub's
hosted runners. Simplest (key-only auth, so still safe):

```bash
gcloud compute firewall-rules create allow-ssh-deploy \
  --allow=tcp:22 --direction=INGRESS --network=default \
  --source-ranges=0.0.0.0/0
```

Tighter (optional): restrict `--source-ranges` to GitHub's Actions IP ranges
from <https://api.github.com/meta> (the `actions` list) — note these rotate, so
you'd need to refresh them periodically.

## 5. Add the repo secrets (GitHub → Settings → Secrets and variables → Actions)

| Secret | Value |
|---|---|
| `VM_HOST` | the VM's public IP |
| `VM_USER` | `ladanibhargav` |
| `VM_SSH_KEY` | contents of the **private** `stockbot_deploy` file |
| `VM_SSH_PORT` | (optional) only if not 22 |

## 6. Test

- Push any commit to `main`, or run the workflow manually:
  **Actions → Deploy to VM → Run workflow**.
- Watch the run log; the final lines should read `OK — stockbot active at <sha>`.

---

## Notes & safety

- **Secrets that stay on the VM** are untouched by deploy: `.env`
  (`MARKETAUX_API_TOKEN`, `ANGEL_*`, `TELETHON_*`, `TELEGRAM_BOT_TOKEN`) and
  `stockagent.db`. They're gitignored, and `git reset --hard` does not remove
  untracked/ignored files — so your keys and data survive every deploy.
- **First deploy of a new key** (e.g. `MARKETAUX_API_TOKEN`): the code arrives
  via deploy, but you must still add the key to `~/stockagent/.env` on the VM by
  hand — deploy will not create it.
- **Immediate restart**: a deploy during market hours (09:15–15:30 IST) briefly
  pauses scanning while the service restarts. To defer daytime restarts to
  after-close, gate `deploy.sh`'s restart on the clock — ask and I'll add it.
- **Rollback**: SSH in and `git reset --hard <old-sha> && bash deploy/deploy.sh`.
- **Security**: this exposes SSH to the internet and stores a private key in
  GitHub. Mitigations baked in: a dedicated key, key-only auth recommended, and
  sudo limited to exactly `systemctl restart/status stockbot`.
