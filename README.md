# Autoencoders & VAEs: teaching a network to imagine cats

An interactive, scrollable web presentation about autoencoders and variational autoencoders, made for MAAI901 Deep Learning for Visual Data. Every key idea is shown live with small models trained on 64x64 cat faces, running right in your browser.

Live at **https://autoencoders.danishali.dev**

## Two halves

- **`training/`** runs once on a desktop GPU: PyTorch dataset loaders, models, training and export. It also precomputes anything too slow for the browser (PCA, latent maps, statistics) and writes the results to `site/models` and `site/data`.
- **`site/`** is a fully static site: plain HTML, CSS and JavaScript modules, no build step and no backend. The exported models run in the viewer's browser with onnxruntime-web, and a service worker keeps it working offline.

See [SPEC.md](SPEC.md) for the full design.
