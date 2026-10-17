# Deploying to Cloudflare Pages

The site is the `site/` folder as it is: no build step. Cloudflare Pages serves it at
https://autoencoders.danishali.dev, with the response headers from `site/_headers`.

## What is already set up in the repo

- **Cache-safe names.** `training/export.py` gives every model and data file a content hash in
  its name (`vae_128.decoder.0fa1da.onnx`). `models/manifest.json` names the model files and
  `data/files.json` maps every logical name (`data/showcase.png`) to its current file. The page
  code asks for logical names and looks them up there. The export then runs `tools/stamp_site.py`.
- **Service worker version.** `tools/stamp_site.py` rewrites `VERSION` in `site/sw.js` from a hash
  of `data/files.json`, `models/manifest.json` and the other site files. It also checks the
  host's limits (every file under 25 MiB, fewer than 20,000 files) and prints the 10 largest files.
- **`site/_headers`**:
  - `/models/*`, `/data/*`, `/vendor/*`: `public, max-age=31536000, immutable`. They never change
    in place: models and data are content-hashed, and the runtime sits in `vendor/ort-1.30.0/`
    (if you upgrade it, use a new folder name).
  - `/css/*`, `/js/*`: `public, max-age=3600`. Hashing these would mean rewriting every
    `import` in the slide modules, which is a build step. An hour is short, and the
    service worker re-fetches the shell past the HTTP cache whenever its version changes.
  - `/`, `/index.html`, `/sw.js`, `/precache.json`, `/models/manifest.json`, `/data/files.json`:
    `no-cache` (always revalidate). Cloudflare joins a header set by two rules with a comma, so
    the two JSON files first remove the folder's long `Cache-Control` (`! Cache-Control`).
  - Everywhere: `Cross-Origin-Opener-Policy: same-origin` and
    `Cross-Origin-Embedder-Policy: require-corp`, which make the page cross-origin isolated, so
    onnxruntime-web runs multi-threaded. Everything the page loads is same-origin; the only
    external URLs are plain links in the credits, which these headers don't affect.
  - `.wasm` as `application/wasm`.
- **`site/404.html`**: Pages serves it for any missing URL.
- **Link previews**: `og:` and `twitter:` tags in `index.html`, image `site/img/og.jpg` (1200x630).

All of this was checked locally with `npx wrangler pages dev site`, which applies `_headers` the
way Pages does.

## Your steps

### 1. Push the repo to GitHub

Create an empty repository named `vae-presentation` on GitHub (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/Danish12196/vae-presentation.git
```

```bash
git push -u origin main
```

### 2. Create the Pages project

Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** tab →
**Import an existing Git repository** → authorise GitHub and pick `vae-presentation`.

| setting | value |
| --- | --- |
| Project name | `cat-vae` (gives `cat-vae.pages.dev`) |
| Production branch | `main` |
| Framework preset | None |
| Build command | *(leave empty)* |
| Build output directory | `site` |

**Save and Deploy**. After about a minute the site is live at `https://cat-vae.pages.dev`
(or whatever name Cloudflare gives the project). Every later push to `main` redeploys it.

### 3. Add the custom domain

In the Pages project → **Custom domains** → **Set up a custom domain** →
`autoencoders.danishali.dev`.

- **If danishali.dev's DNS is on Cloudflare**, it adds the record for you. Confirm, and you're done.
- **If not**, add this record at your DNS provider:

  | type | name | target |
  | --- | --- | --- |
  | CNAME | `autoencoders` | `cat-vae.pages.dev` (your project's `*.pages.dev` name) |

  Then wait for Cloudflare to verify it and issue the certificate. This takes minutes, sometimes
  a few hours. The domain shows **Active** when it's done.

### Optional: deploy without the dashboard

Log in once, then deploy straight from your machine:

```bash
npx wrangler login
```

```bash
npx wrangler pages deploy site --project-name cat-vae
```

The first run creates the project. Add the custom domain in the dashboard as in step 3. With
this route, pushing to GitHub doesn't deploy; run the command again after each change.

## After changing the site

1. If you re-trained models: `python training/export.py` (it re-stamps the site).
   Otherwise, after editing anything in `site/`: `python tools/stamp_site.py`.
2. Commit and push. Pages redeploys. Visitors get the new version on their next visit (the service
   worker sees a new `VERSION` and swaps its cache).

## Checking the live site

```bash
curl -sI https://autoencoders.danishali.dev/
```

Look for `cache-control: no-cache` and the two `cross-origin-*` headers. A `.onnx` file should show
`public, max-age=31536000, immutable`; the `.wasm` file should also show `content-type: application/wasm`.
