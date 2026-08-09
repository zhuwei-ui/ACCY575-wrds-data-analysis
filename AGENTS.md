# WRDS Cloud access

Reach the WRDS Cloud **only** through the multiplexed SSH host alias `wrds`:

The remote working directory is `~/accy575`.

- Run a remote command:  `ssh wrds '<command>'`
- Sync code up:          `rsync -avz --delete --exclude '.venv' --exclude '__pycache__' \
                            --exclude '.git' --exclude 'data' --exclude 'logs' ./ wrds:~/accy575/`
- Pull results down:     `rsync -avz wrds:~/accy575/data/raw/ ./data/raw/`

Rules:
- NEVER use `ssh <user>@wrds-cloud.wharton.upenn.edu` — that endpoint is
  password + Duo only and will hang waiting for human input.
- NEVER start a long-running or interactive process on the login node.
  The login node is shared and process-capped. Heavy work goes to the
  compute grid via `qsub`.
- NEVER drop `--exclude 'data'` or `--exclude 'logs'` from the upward sync.
  Combined with `--delete` that erases results produced on the cloud, and
  `~/accy575/data` is a symlink to scratch — deleting it strands the data.
- NEVER add `--delete` to the downward sync.
- Home on WRDS has a 10 GB quota. Data output goes to `~/accy575/data`,
  which is a symlink into /scratch. Never write large files elsewhere.
- Edit all files locally. The copy on WRDS is a disposable working copy,
  never the source of truth.
- Do not `git push` from WRDS. The repo lives on the laptop.
- Anything submitted with `qsub` runs unattended: no prompts, no Duo, no
  `breakpoint()`, no Postgres connection. Log to stdout instead.
- If a command hangs, check `ssh -O check wrds` before anything else.
