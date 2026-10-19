# Deploying to Vercel

The site is the `site/` folder as it is: no build step. Vercel serves it at
https://autoencoders.danishali.dev, with the response headers from `vercel.json` (repo root).
danishali.dev's DNS is already on Vercel (team "Danish's projects"), so the subdomain needs no
manual DNS record.

## What is already set up in the repo

- **`vercel.json`**: no framework, no build, output directory `site`, plus the headers:
  - `/models/*`, `/data/*`, `/vendor/*`: `public, max-age=31536000, immutable`. They never change
    in place: models and data have a content hash in their names, and the runtime sits in
    `vendor/ort-1.30.0/` (if you upgrade it, use a new folder name).
  - `/css/*`, `/js/*`: `public, max-age=3600`. Hashing these would mean rewriting every
    `import` in the slide modules, which is a build step. An hour is short, and the
    service worker re-fetches the shell past the HTTP cache whenever its version changes.
  - `/`, `/index.html`, `/sw.js`, `/precache.json`, `/models/manifest.json`, `/data/files.json`:
    `no-cache` (always revalidate). The folder rules use a regex that leaves these two JSON files
    out, so every path gets exactly one `Cache-Control`.
  - Everywhere: `Cross-Origin-Opener-Policy: same-origin` and
    `Cross-Origin-Embedder-Policy: require-corp`, which make the page cross-origin isolated, so
    onnxruntime-web runs multi-threaded. Everything the page loads is same-origin; the only
    external URLs are plain links in the credits, which these headers don't affect.
  - `.wasm` as `application/wasm`.
- **`.vercelignore`** keeps `training/`, `tools/`, `docs/` and `design-mockups/` out of CLI uploads. The mockups are design studies, not part of the site. The service worker only precaches files under `site/`.
- **Cache-safe names.** `training/export.py` gives every model and data file a content hash in
  its name (`vae_128.decoder.0fa1da.onnx`). `models/manifest.json` names the model files and
  `data/files.json` maps every logical name (`data/showcase.png`) to its current file. Then it runs
  `tools/stamp_site.py`, which sets the service worker's cache version from a hash of
  `files.json`, `manifest.json` and the other site files. It also checks the limits (every file
  under 25 MiB, at most 15,000 files) and prints the 10 largest files.
- **`site/404.html`**: Vercel serves it for any missing URL. Once the service worker is installed it does the same, online and offline, and `stamp_site.py` includes the page in the shell cache.
- **Link previews**: `og:` and `twitter:` tags in `index.html`, image `site/img/og.jpg` (1200x630).

## Your steps

### 1. Push the repo to GitHub

Create an empty repository named `vae-presentation` on GitHub (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/Danish12196/vae-presentation.git
```

```bash
git push -u origin main
```

### 2. Import it into Vercel

vercel.com → **Add New…** → **Project** → pick `vae-presentation` (install the Vercel GitHub app
for the repo if it isn't listed). Team: **Danish's projects**.

| setting | value |
| --- | --- |
| Project name | `vae-presentation` |
| Framework preset | Other |
| Root directory | `./` |
| Build and output settings | leave as is (`vercel.json` sets output `site`, no build) |

**Deploy**. After about a minute it is live at `https://vae-presentation.vercel.app`. Every push to `main`
redeploys production; other branches get preview URLs.

### 3. Add the custom domain

Project → **Settings** → **Domains** → **Add** → `autoencoders.danishali.dev`. danishali.dev uses
Vercel's nameservers, so Vercel creates the record and the certificate itself; it shows
**Valid Configuration** within a minute or two. (Claude can also do this step for you through
the Vercel connector once the project exists.)

### Alternative: deploy from your machine with the CLI

Log in once, then from the repo root:

```bash
npx vercel login
```

```bash
npx vercel --prod
```

The first run asks for the scope (Danish's projects) and project name (`vae-presentation`) and links the
folder (it writes `.vercel/`, which is git-ignored). Then add the domain:

```bash
npx vercel domains add autoencoders.danishali.dev
```

With this route, pushing to GitHub doesn't deploy; run `npx vercel --prod` after each change.

## After changing the site

1. If you re-trained models: `python training/export.py` (it re-stamps the site).
   Otherwise, after editing anything in `site/`: `python tools/stamp_site.py`.
2. Commit and push. Vercel redeploys. Visitors get the new version on their next visit (the
   service worker sees a new `VERSION` and swaps its cache).

## Notes

- **Preview deployments** inject the Vercel Toolbar from vercel.live. Cross-origin isolation blocks
  it, so previews may log a console error about it. Production has no toolbar.
- **Checking the live site**:

  ```bash
  curl -sI https://autoencoders.danishali.dev/
  ```

  Look for `cache-control: no-cache` and the two `cross-origin-*` headers. A `.onnx` file should show
  `public, max-age=31536000, immutable`; the `.wasm` file also `content-type: application/wasm`.
