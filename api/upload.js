const DEFAULT_MAX_SAMPLE_BYTES = 10 * 1024 * 1024;

module.exports = async function uploadHandler(req, res) {
  setCorsHeaders(res);

  if (req.method === "OPTIONS") {
    return send(res, 204, {});
  }

  if (req.method !== "POST") {
    return send(res, 405, { ok: false, error: "Method not allowed" });
  }

  const config = readConfig();
  if (!config.ok) {
    return send(res, 500, { ok: false, error: config.error });
  }

  let body;
  try {
    body = await readJsonBody(req, config.maxSampleBytes * 2);
  } catch (error) {
    return send(res, error.statusCode || 400, { ok: false, error: error.message });
  }

  if (config.uploadSecret) {
    const supplied = req.headers["x-upload-secret"] || body.uploadKey || "";
    if (supplied !== config.uploadSecret) {
      return send(res, 401, { ok: false, error: "Invalid upload key" });
    }
  }

  try {
    const sample = buildSampleFiles(body, config);
    if (sample.totalBytes > config.maxSampleBytes) {
      return send(res, 413, {
        ok: false,
        error: `Sample is too large. Limit is ${Math.round(config.maxSampleBytes / 1024 / 1024)} MB.`,
      });
    }

    const commit = await commitFiles(config, sample.files, sample.commitMessage);
    return send(res, 200, {
      ok: true,
      commit: commit.sha,
      commit_url: commit.html_url,
      paths: sample.files.map((file) => file.path),
    });
  } catch (error) {
    const status = error.statusCode && error.statusCode < 500 ? error.statusCode : 500;
    return send(res, status, { ok: false, error: error.message });
  }
};

function readConfig() {
  const githubToken = process.env.GITHUB_TOKEN;
  const githubRepo = process.env.GITHUB_REPO;
  if (!githubToken) {
    return { ok: false, error: "Missing GITHUB_TOKEN environment variable" };
  }
  if (!githubRepo || !/^[^/\s]+\/[^/\s]+$/.test(githubRepo)) {
    return { ok: false, error: "Missing or invalid GITHUB_REPO environment variable" };
  }

  return {
    ok: true,
    githubToken,
    githubRepo,
    branch: process.env.GITHUB_BRANCH || "master",
    outputDir: trimSlashes(process.env.GITHUB_OUTPUT_DIR || "output"),
    uploadSecret: process.env.UPLOAD_SECRET || "",
    allowedOrigin: process.env.ALLOWED_ORIGIN || "",
    maxSampleBytes: Number(process.env.MAX_SAMPLE_BYTES || DEFAULT_MAX_SAMPLE_BYTES),
  };
}

async function readJsonBody(req, maxBytes) {
  if (Buffer.isBuffer(req.body)) {
    return JSON.parse(req.body.toString("utf8"));
  }
  if (req.body && typeof req.body === "object") {
    return req.body;
  }
  if (typeof req.body === "string") {
    return JSON.parse(req.body);
  }

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) {
      const error = new Error("Request body is too large");
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }

  if (!chunks.length) {
    return {};
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function buildSampleFiles(body, config) {
  const username = safe(body.username || body.metadata?.username || "anonymous");
  const label = safe(body.label || body.metadata?.label || "unknown");
  const sampleId = safe(body.sampleId || makeServerSampleId());
  const baseDir = [config.outputDir, username, label].filter(Boolean).join("/");
  const baseName = `${label}_${sampleId}`;
  const image = decodeDataUrl(body.image, "image/");
  const video = body.video ? decodeDataUrl(body.video, "video/") : null;

  if (!image && !video) {
    const error = new Error("No image or video was provided");
    error.statusCode = 400;
    throw error;
  }

  const imagePath = image ? `${baseDir}/${baseName}.png` : null;
  const videoExt = safeExtension(body.video?.extension || extensionFromMime(video?.mime) || "webm");
  const videoPath = video ? `${baseDir}/${baseName}.${videoExt}` : null;
  const jsonPath = `${baseDir}/${baseName}.json`;

  const metadata = {
    ...(body.metadata || {}),
    username,
    label,
    sample_id: sampleId,
    created_at: body.createdAt || body.metadata?.created_at || new Date().toISOString(),
    repository: {
      repo: config.githubRepo,
      branch: config.branch,
      paths: {
        image: imagePath,
        video: videoPath,
        json: jsonPath,
      },
    },
    uploaded_at: new Date().toISOString(),
  };

  const files = [];
  let totalBytes = 0;

  if (image) {
    files.push({ path: imagePath, contentBase64: image.base64 });
    totalBytes += image.byteLength;
  }
  if (video) {
    files.push({ path: videoPath, contentBase64: video.base64 });
    totalBytes += video.byteLength;
  }

  const jsonBase64 = Buffer.from(JSON.stringify(metadata, null, 2), "utf8").toString("base64");
  files.push({ path: jsonPath, contentBase64: jsonBase64 });
  totalBytes += Buffer.byteLength(jsonBase64, "base64");

  return {
    files,
    totalBytes,
    commitMessage: `Add airwriting sample ${username}/${label}/${sampleId}`,
  };
}

function decodeDataUrl(file, expectedPrefix) {
  if (!file || typeof file.dataUrl !== "string") {
    return null;
  }
  // MIME may carry parameters, e.g. "video/webm;codecs=vp9" from MediaRecorder.
  const match = file.dataUrl.match(/^data:(.+?);base64,([A-Za-z0-9+/=\s]+)$/);
  if (!match) {
    const error = new Error("Invalid data URL");
    error.statusCode = 400;
    throw error;
  }
  const mime = match[1];
  if (!mime.startsWith(expectedPrefix)) {
    const error = new Error(`Unexpected media type: ${mime}`);
    error.statusCode = 400;
    throw error;
  }
  const base64 = match[2].replace(/\s/g, "");
  return {
    mime,
    base64,
    byteLength: Buffer.byteLength(base64, "base64"),
  };
}

async function commitFiles(config, files, message) {
  let lastError;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const ref = await github(config, "GET", `git/ref/heads/${config.branch}`);
      const baseCommit = await github(config, "GET", `git/commits/${ref.object.sha}`);
      const blobs = await Promise.all(files.map(async (file) => {
        const blob = await github(config, "POST", "git/blobs", {
          content: file.contentBase64,
          encoding: "base64",
        });
        return { ...file, sha: blob.sha };
      }));
      const tree = await github(config, "POST", "git/trees", {
        base_tree: baseCommit.tree.sha,
        tree: blobs.map((file) => ({
          path: file.path,
          mode: "100644",
          type: "blob",
          sha: file.sha,
        })),
      });
      const commit = await github(config, "POST", "git/commits", {
        message,
        tree: tree.sha,
        parents: [ref.object.sha],
      });
      await github(config, "PATCH", `git/refs/heads/${config.branch}`, {
        sha: commit.sha,
        force: false,
      });
      return commit;
    } catch (error) {
      if (error.statusCode === 404 && error.githubPath === `git/ref/heads/${config.branch}`) {
        throw branchNotFoundError(config.branch);
      }
      lastError = error;
      if (![409, 422].includes(error.statusCode) || attempt === 1) {
        throw error;
      }
    }
  }
  throw lastError;
}

async function github(config, method, path, body) {
  const response = await fetch(`https://api.github.com/repos/${config.githubRepo}/${path}`, {
    method,
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${config.githubToken}`,
      "Content-Type": "application/json",
      "User-Agent": "airwriting-uploader",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (error) {
      data = { message: text };
    }
  }

  if (!response.ok) {
    const error = new Error(data.message || `GitHub API error ${response.status}`);
    error.statusCode = response.status;
    error.githubPath = path;
    throw error;
  }
  return data;
}

function branchNotFoundError(branch) {
  const error = new Error(`GitHub branch "${branch}" was not found. Set GITHUB_BRANCH to the branch Vercel deploys from, such as "master".`);
  error.statusCode = 400;
  return error;
}

function safe(value) {
  const text = String(value || "").trim().normalize("NFC");
  return text
    .replace(/[^\p{L}\p{M}\p{N}_.-]/gu, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "") || "x";
}

function safeExtension(value) {
  const ext = safe(value).toLowerCase();
  return ext === "mp4" ? "mp4" : "webm";
}

function extensionFromMime(mime) {
  if (!mime) {
    return "";
  }
  return mime.includes("mp4") ? "mp4" : "webm";
}

function makeServerSampleId() {
  const d = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}_${Math.random().toString(36).slice(2, 8)}`;
}

function trimSlashes(value) {
  return String(value || "").replace(/^\/+|\/+$/g, "");
}

function setCorsHeaders(res) {
  if (!process.env.ALLOWED_ORIGIN) {
    return;
  }
  res.setHeader("Access-Control-Allow-Origin", process.env.ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, x-upload-secret");
}

function send(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(status === 204 ? "" : JSON.stringify(payload));
}
