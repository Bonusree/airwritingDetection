# Air Writing Collector

Air Writing Collector lets users draw in the air with a webcam, then saves a trajectory image, recorded video, and JSON feature metadata.

Users open a website on Windows, Linux, Android, or any modern browser. They do not install Python, Flask, OpenCV, or MediaPipe.

## What Gets Saved

Every saved sample is committed to your GitHub repository under:

```text
output/{user}/{label}/
```

Each sample contains:

- `{label}_{sample_id}.png` - trajectory image
- `{label}_{sample_id}.webm` or `{label}_{sample_id}.mp4` - browser-recorded video
- `{label}_{sample_id}.json` - metadata and 8-D trajectory features

The exact video format depends on the user's browser.
The app label menu uses Bangla vowels: `অ`, `আ`, `ই`, `ঈ`, `উ`, `ঊ`, `ঋ`, `এ`, `ঐ`, `ও`, `ঔ`.

## Deploy For Public Users

Use Vercel or another host that supports static files plus Node serverless functions. GitHub Pages alone cannot safely save files into your repo because it cannot hide a GitHub write token.

### 1. Push This Repo To GitHub

Commit the project, including:

```text
public/
api/upload.js
vercel.json
package.json
```

### 2. Create A GitHub Token

Create a fine-grained GitHub personal access token for the target repository.

Required permission:

```text
Contents: Read and write
```

Keep this token private. Never put it in `public/app.js`, HTML, or any browser code.

### 3. Deploy On Vercel

1. Go to Vercel.
2. Import this GitHub repository.
3. In project settings, use:

```text
Framework Preset: Other
Root Directory: ./
Build Command: empty
Output Directory: empty
Install Command: empty
```

4. Add these environment variables:

```text
GITHUB_TOKEN=your_fine_grained_github_token
GITHUB_REPO=your-github-username/airwriting
GITHUB_BRANCH=master
GITHUB_OUTPUT_DIR=output
```

Recommended optional protection:

```text
UPLOAD_SECRET=choose-a-shared-upload-key
```

If `UPLOAD_SECRET` is set, users must enter that key in the app before uploads work. If you leave it empty, anyone with the site URL can upload samples to your repo.

### What Is The Upload Key?

The upload key is **not** your GitHub token.

- `GITHUB_TOKEN` is private and stays only inside Vercel environment variables.
- `UPLOAD_SECRET` is an optional shared password for app users.
- If `UPLOAD_SECRET=abc123`, users type `abc123` into the app's Upload key field.
- If `UPLOAD_SECRET` is empty, leave the app's Upload key field empty.

5. Deploy.
6. Open the Vercel URL on desktop or Android and allow camera access.
7. Use the browser install button or the browser's install/add-to-home-screen menu.

## Local PWA Development

Install the Vercel CLI if you want to test the deployed app locally:

```bash
npx vercel dev
```

Create a local `.env` from `.env.example` before testing uploads:

```bash
cp .env.example .env
```

Then open the local URL printed by Vercel.

Do not use `python3 -m http.server` to test uploads. It can preview the static UI, but it cannot run `api/upload.js`, so saving will fail.

## Verify GitHub Saving

After writing a sample in the deployed app:

1. The app should show `Uploaded`.
2. Open your GitHub repo in the browser.
3. Check `output/{user}/{label}/`.
4. You should see one `.png`, one `.webm` or `.mp4`, and one `.json` for the same sample.
5. Check the repo commit history for a commit named like `Add airwriting sample user1/A/...`.

If it does not save, check the Vercel function logs for `/api/upload`. Most failures are one of these:

- `Missing GITHUB_TOKEN environment variable`
- `Invalid upload key`
- GitHub token does not have `Contents: Read and write`
- Sample video is larger than `MAX_SAMPLE_BYTES`

## Updating The Deployed App

If this repository is connected to Vercel, a local commit only updates the deployed app after you push it to the branch Vercel deploys from. For example:

```bash
git add .env.example README.md api/upload.js public/index.html public/app.js public/styles.css public/sw.js
git commit -m "Fix app controls"
git push origin master
```

Vercel then creates a new deployment automatically, unless automatic deployments are disabled. If you deployed manually, run `npm run deploy` instead.

## Delete Saved Samples

To delete a bad sample from GitHub:

1. Open `output/{user}/{label}/` in your GitHub repo.
2. Delete the matching `.png`, video, and `.json` files.
3. Commit the deletion.

From the command line:

```bash
git rm output/user1/A/A_sampleid.png output/user1/A/A_sampleid.webm output/user1/A/A_sampleid.json
git commit -m "Delete bad airwriting sample"
git push
```

## Project Structure

```text
airwriting/
├── api/upload.js                 # Serverless GitHub uploader
├── public/                       # Browser/PWA app
│   ├── app.js
│   ├── index.html
│   ├── manifest.webmanifest
│   ├── models/hand_landmarker.task
│   └── styles.css
├── package.json                  # Vercel scripts
└── vercel.json                   # Deployment routing
```

## Notes

- Camera access requires HTTPS on phones. Vercel provides HTTPS automatically.
- GitHub is not ideal for very large videos. The browser app records short, low-bitrate clips and auto-saves after a pause.
- If collected videos may contain private faces, rooms, or names, use a private GitHub repository.
