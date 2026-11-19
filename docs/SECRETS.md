# Secrets Reference

The ingestion connectors only reach their upstream APIs when the expected secrets are present. Export these values locally (for
example via `direnv` or a `.env` file) and configure the same names under **Settings → Secrets and variables → Actions** when
running in GitHub Actions.

| Secret | Used by | Notes |
|---|---|---|
| `DTM_API_KEY` | `resolver.ingestion.dtm_client` | Single IOM DTM subscription key passed to the official `dtmapi` SDK. Real runs exit with code `2` if the key is missing or invalid; use `--offline-smoke` / `DTM_OFFLINE_SMOKE=1` for header-only diagnostics without credentials. |

Remove any legacy references to `DTM_API_PRIMARY_KEY`, `DTM_API_SECONDARY_KEY`, or `DTM_SUBSCRIPTION_KEY`; the connector now reads only `DTM_API_KEY`.

## Google Drive archive (`RCLONE_CONF_B64`)

The **Archive Run to Drive** workflow (`.github/workflows/archive_to_drive.yml`) copies every finished chain into a `Fred_archive` folder at the root of your Google Drive: one zip per chain, loose summaries beside it, and a compressed DB snapshot after each forecast chain. It reaches Drive with [rclone](https://rclone.org), signed in as you, with the `drive.file` scope. That scope lets rclone see only the files and folders rclone itself created, nothing else in your Drive.

| Secret | Used by | Notes |
|---|---|---|
| `RCLONE_CONF_B64` | `archive_to_drive.yml` | Base64 of an `rclone.conf` holding exactly one remote, named `pythia_drive`. Decoded to a file in one step of each job and deleted at the end of the job. |

### One-time setup (Windows, PowerShell)

1. **Create a Google Cloud project and enable the Drive API.** In the [Google Cloud console](https://console.cloud.google.com/), create a project (any name, for example `fred-archive`). Open **APIs & Services → Library**, search for **Google Drive API** and click **Enable**.

2. **Configure the OAuth consent screen, then publish it.** Under **APIs & Services → OAuth consent screen** (shown as **Google Auth Platform → Branding / Audience** in newer consoles), choose **External**, give the app a name and your email, and add yourself under **Test users**. Then, under **Audience**, click **Publish app** so the status reads **In production**.
   Why: Google expires the refresh token of an app left in **Testing** after 7 days, and the archive would stop silently a week later. The `drive.file` scope is not a sensitive or restricted scope, so a published app that uses only it needs no Google verification; the consent screen may say the app is unverified, which is expected for your own app.

3. **Create a Desktop OAuth client and an rclone remote.** Under **APIs & Services → Credentials → Create credentials → OAuth client ID**, choose **Desktop app**. Copy the client ID and client secret. Then, in PowerShell:

   ```powershell
   winget install Rclone.Rclone
   ```

   Close and reopen PowerShell so `rclone` is on the path. Use a config file of its own, so no other rclone remote ends up in the secret:

   ```powershell
   $conf = "$env:USERPROFILE\pythia-rclone.conf"
   ```

   ```powershell
   rclone --config $conf config create pythia_drive drive client_id=PASTE_CLIENT_ID client_secret=PASTE_CLIENT_SECRET scope=drive.file
   ```

   A browser opens; sign in with the Google account whose Drive should hold the archive and allow access. Check the remote works (it lists nothing yet, which is right):

   ```powershell
   rclone --config $conf lsd pythia_drive:
   ```

4. **Save the config as a repository secret.** Base64-encode it and copy it to the clipboard:

   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes($conf)) | Set-Clipboard
   ```

   In GitHub, open **Settings → Secrets and variables → Actions → New repository secret**, name it `RCLONE_CONF_B64` and paste. Then delete the local file, which holds a token to your Drive:

   ```powershell
   Remove-Item $conf
   ```

5. **Remove any hand-made `Fred_archive` folder.** With the `drive.file` scope rclone cannot see a folder you made in the Drive web page, so it would create a second `Fred_archive` beside it. If you created one, delete it in Drive and empty it from **Trash**. The first archive run creates `Fred_archive` at the Drive root. If rclone ever sees two folders of that name, the run fails with an error and uploads nothing.

6. **Archive what GitHub still holds.** In GitHub, open **Actions → Archive Run to Drive → Run workflow**, set `backfill_days` to `80`, and run it. From then on, every Publish run dispatches the archive itself, and a weekly run catches any that were missed.

The daily **Cron Watchdog** checks that every Publish run older than 48 hours has been archived, and that Drive has at least 20 GB free and is under 85% full.
