# Autoencoders & VAEs: teaching a network to imagine cats

An interactive, scrollable web presentation about autoencoders and variational autoencoders, made for MAAI901 Deep Learning for Visual Data. Every key idea is shown live with small models trained on 64x64 cat faces, running right in your browser.

Live at **https://autoencoders.danishali.dev**

## Two halves

- **`training/`** runs once on a desktop GPU: PyTorch dataset loaders, models, training and export. It also precomputes anything too slow for the browser (PCA, latent maps, statistics) and writes the results to `site/models` and `site/data`.
- **`site/`** is a fully static site: plain HTML, CSS and JavaScript modules, no build step and no backend. The exported models run in the viewer's browser with onnxruntime-web, and a service worker keeps it working offline.

See [SPEC.md](SPEC.md) for the full design.

## Run it locally

```bash
python tools/serve.py
```

Then open http://localhost:8000. Any static file server works (there is no build step), e.g.
`python -m http.server 8000 --directory site`, but `tools/serve.py` turns off browser caching
so edited slide modules show up on a normal reload.
The offline service worker is off on localhost so your edits show up on reload; open
http://localhost:8000/?sw=on to test offline mode, and `?sw=off` to turn it off again.

After changing anything in `site/` (or re-running `training/export.py`), run
`python tools/stamp_site.py` so the service worker caches the new files.
