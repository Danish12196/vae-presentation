# Cat VAE presentation: spec

## Goal
A 25-minute class presentation (MAAI901 Deep Learning for Visual Data) on autoencoders and VAEs, built as a single scrolling web page instead of slides. Every key idea is demonstrated live with small models trained on 64x64 cat faces. The script lives in docs/script.md. After class, the same site stays public at https://autoencoders.danishali.dev for anyone to explore.

## Setting
- Presented from any device with a modern browser: a laptop on a projector, a tablet or a phone. Must look right at 1920x1080 and 1366x768, and stay usable on a phone in portrait (layouts stack vertically).
- Speaker notes are NOT part of the site (the presenter reads them from the script doc on a phone).
- Tone: fun, colourful, a bit of humour, still appropriate for a master's class.

## Architecture
- training/ (desktop, RTX 4070 Ti, run once): PyTorch dataset loaders, models, train.py and export.py. All heavy work happens here: training, and precomputing anything slow (PCA, latent maps, classifier visualisations, statistics). Exports go to site/models and site/data.
- Training outputs: training/runs/<name>/model.pt (run config, state_dict, per-epoch loss history) plus epoch_XXX.png previews; cnn_cls also writes metrics.json. `--quick` smoke tests write to training/runs/quick/ so they never count as finished runs for `--skip-existing`.
- site/: fully static HTML, CSS and vanilla JavaScript ES modules. No build step, no framework, no backend. onnxruntime-web (1.30, the single-file WASM bundle) is vendored into site/vendor and runs the small models in the viewer's browser. WASM only: the WebGPU builds need a 26.8 MB .wasm, over the 25 MB file limit, and WASM already runs each of our models in a few milliseconds. Single-threaded unless the page is cross-origin isolated. Fonts are vendored woff2 files (Latin plus Greek for the maths).
- The browser only runs small encoders, decoders and the classifier on 64x64 images. Each call should take under ~50 ms on a mid-range phone; anything slower is precomputed.
- A service worker (site/sw.js) caches every file after the first visit, so the site keeps working offline (classroom Wi-Fi can fail). It precaches the shell on install; models, data and the runtime are cached as the loading screen downloads them; everything is served cache-first. tools/stamp_site.py writes site/precache.json (every file and its size) and the cache VERSION in sw.js; run it after any change in site/. On localhost the service worker is off unless you open the page with ?sw=on (and ?sw=off turns it off again).
- Shell (site/js): main.js (navigation, top bar, overlays, lazy slide modules), loader.js (first-visit downloads and model warm-up), offline.js, models.js (sessions, encode/decode/classify, tensors, sprites, ModelMissing and the "still training" card), ui.js (shared widgets), mascot.js. Each slide's demo lives in js/slides/sNN.js exporting init(sectionEl) and optional enter()/leave(); main.js imports it when the slide comes within one screen. Elements that a finger drags inside a demo get class="interactive" so swipes starting there don't change slides. Shared demo pieces: js/viz.js (error heatmap and histogram), js/state.js (the image picked on s02, reused by s03 and s04), js/netdiagram.js (the layer diagram with real shapes, used by s03 and s04). Slide CSS lives in css/partN.css. tools/serve.py serves site/ locally with caching off.
- Size budget: total download under 60 MB as transferred (gzip), every single file under 25 MB (static host limits). Today: about 58 MB transferred, 72.8 MB on disk (the 14.2 MB runtime .wasm gzips to 3.7 MB). export.py stores the biggest models' weights as float16 (cast back to float32 inside the graph, so every onnxruntime-web backend runs them) until everything fits, but only where the float16 file still matches PyTorch within 1e-2.
- Exports (training/export.py): site/models/<name>.encoder.onnx and .decoder.onnx (cnn_cls.onnx for the classifier), opset 17, dynamic batch; site/models/manifest.json; site/data/ holds the showcase sprite (the 64 held-out demo cats), animals, cls_dream, PCA reconstructions, latent maps (thumbnails as JPEG: 1 MB instead of 4.7 MB as PNG), anomaly statistics, not-cats, sparse codes, and files.json listing every file with its size.
- site/models/manifest.json lists every model, its latent size and files. Any demo whose model is missing shows a friendly "this cat is still training" card instead of breaking.

## Datasets
- Cats: 29,800 cat faces, 64x64 RGB, from https://github.com/fferlito/Cat-faces-dataset (three .tar.gz parts, about 90 MB each, from raw.githubusercontent.com). Hold out 512 fixed cats (seed 0) that are never trained on; the site uses them as demo inputs.
- Classifier only: AFHQ (Animal Faces-HQ: cat, dog, wild) resized to 64x64, downloaded on the desktop from Hugging Face (e.g. huggan/AFHQv2). If unavailable, use Oxford-IIIT Pet head crops (cat vs dog). The classifier's cat class mixes AFHQ cats with fferlito cats, so it cannot learn "which dataset" instead of "which animal".

## Models
All share one conv trunk: conv 32-64-128-256 (stride 2, BatchNorm, LeakyReLU) down to 256x4x4, flattened to 4096. Autoencoders add a linear layer to the latent and a mirrored decoder (transposed convs, sigmoid). The classifier adds a linear head instead of a decoder. Teaching point: classifier = encoder + label head; autoencoder = encoder + decoder. Reconstruction loss = summed squared error per image.
| name | type | latent | used for |
| cnn_cls | CNN classifier, 3 classes (cat, dog, wild) | - | live classification (s02, s03), "ask the classifier to draw a cat" (s02), judging generated cats (s10, s15) |
| ae_2, ae_16, ae_64, ae_256 | plain AE | 2/16/64/256 | bottleneck slider; ae_2 also draws the Part 4 latent map; ae_64 runs the anomaly detector |
| dae_128 | denoising AE (input noise sigma 0 to 0.5, clean target) | 128 | denoising demo |
| sparse_l1 | ReLU code + L1 penalty | 1024 | sparse demo (codes, exact % active and reconstructions precomputed for the 64 showcase cats; the two sparse models are 39 MB, too big to ship) |
| sparse_none | same, no penalty | 1024 | sparse comparison (precomputed, as above) |
| cae_64 | contractive (penalise code change under small input noise, relative to the code spread; weight 100) | 64 | contractive demo, compared with ae_64 |
| vae_2d_b0, vae_2d_b0.1, vae_2d_b1, vae_2d_b4 | VAE, beta 0/0.1/1/4 | 2 | KL slider; vae_2d_b1 also draws the Part 5 map |
| vae_128 | VAE, beta 1 | 128 | generation, interpolation, reparameterization |
| ae_128 | plain AE | 128 | AE vs VAE sampling and interpolation |
VAEs use KL warm-up over the first 5 epochs.

## Slides (one full-screen section each)
Part 1: From classification to generation
1. Title: "Autoencoders & VAEs: teaching a network to imagine cats". Animated cat mascot.
2. Discriminative vs generative: the real cnn_cls classifies a picked image (cats plus a few held-out dogs and wild animals) with live probability bars. A flip switch reverses the arrow: "cat" -> a vae_128 cat. A button "ask the classifier to draw a cat" shows the precomputed gradient-ascent image: a psychedelic mess. It knows a cat when it sees one, but can't draw one.
3. Encoder-decoder: layers build up one by one with real tensor shapes; hover a layer for its shape. A toggle swaps the end of the network: "classifier head" (outputs a label) vs "decoder" (outputs an image), showing they share the same encoder.
4. The identity trap: toggle "make every layer as wide as the input". The network becomes "Photocopier Cat": loss 0.000, features learned: none.
Part 2: Autoencoders
5. Bottleneck: live slider over ae_2/16/64/256 on a held-out cat, showing 12,288 numbers -> k numbers and the compression ratio. Pick any of 8 held-out cats.
6. Reconstruction loss: input, output and a pixel heatmap of squared error; live MSE and BCE values; a counter "labels used: 0".
7. PCA vs autoencoder: the same cats reconstructed by PCA and by the AE at the same k, side by side, plus the average error over all 64 showcase cats (four cats can go either way; on average the AE wins at every k). Callout: a linear AE with MSE learns the PCA subspace.
8. Uses: four cards (compression, denoising, feature extraction, anomaly detection). The anomaly card is live: draw on a cat or pick a "not a cat" image; reconstruction error is placed on a histogram of normal cats (lines at the 1st, 95th and 99th percentiles) and a verdict chip flags both sides: above the 99th "ANOMALY: can't rebuild it" (noise, stripes), between the 95th and 99th "hmm, suspicious", below the 1st "ANOMALY: suspiciously easy" (the beige wall, the fur zoom). The picker's cats are typical held-out cats (a few real cats score above the 99th percentile: false alarms). The upside-down and colour-inverted cats pass as normal cats; the card shows them as the honest blind spot (low reconstruction error means "looks like cat-ish fur and faces", not "a proper cat").
Part 3: Other types of autoencoders
9. Four tabs. Sparse: 1024 neurons as a 32x32 grid lighting up, with a penalty on/off toggle and "% of neurons on". Denoising: a noise slider, noisy input -> clean output. Denoising compares dae_128 with ae_128 on the same noisy cat (the DAE clearly wins from sigma ~0.25 up). Contractive: a "jiggle" slider; about 4 times a second the input is re-jiggled and bars show how far the code moves for ae_64 vs cae_64 (distance divided by the mean of each model's code_std from manifest.json). Convolutional: recap of the architecture we have been using (dae_128 is a convolutional denoising AE). One shared cat picker for all four tabs.
Part 4: Why autoencoders can't generate
10. The ae_2 latent space as a scatter of training cats, each dot coloured by that cat's average fur colour (js/latentmap.js, shared with Part 5; the view is 2x the data's 1st-99th percentile range). Click anywhere to decode that point live with ae_2. The verdict is about the map, not the picture: "seen territory" if the nearest training cat is within the 95th percentile of the cats' own nearest-neighbour distances, else "here be monsters", with the distance in units of the code spread. A "random point" button keeps a tally "in the data / in the gaps". Old-map "here be monsters" labels mark the empty regions. Not cnn_cls: it rates every decoder output as "cat 100%" (even flat green noise), and ae_2's gaps decode to washed-out ghost cats rather than garbage.
Part 5: Variational autoencoders
11. Core idea: encode a cat with vae_2d_b1, draw its mu and sigma as a glowing cloud on the map, a "sample again" button decodes different z from the same cloud (last 6 samples kept in a row). The clouds are tiny next to the whole space (sigma ~0.02 to 0.1 on a map spanning about ±4), so after picking a cat the map zooms smoothly to ~12 sigma around mu and the readout says by how much.
12. Loss = reconstruction + KL: draggable Gaussian (mu, sigma) against N(0,1) with the KL value updating live.
13. KL slider: beta 0 -> 0.1 -> 1 -> 4 animates between the four precomputed latent scatters, from scattered clusters to packed around zero. ELBO callout.
14. Reparameterization trick: a computation graph (left to right on wide screens, top to bottom on phones). "Naive sampling" shows the gradient blocked at the sampling node; "trick" shows gradients flowing through mu and sigma, with epsilon as an outside input. mu/sigma sliders and a "resample epsilon" button drive a live decoded cat.
15. Generating: a grid of 24 cats sampled from vae_128 (re-roll button) beside 24 samples from ae_128 with a toggle: naive z ~ N(0,1) (flat green grass) or fair z matched to the AE's own code mean/std (collages). No classifier scores: cnn_cls says "cat ~100%" for every decoder output, grass included, so the images speak for themselves. The vae_2d_b1 map where any click gives a cat.
16. Interpolation: pick two cats (A and B pickers), scrub between them; AE row vs VAE row of 8 frames each, plus a big live frame per model. Honest note: linear interpolation between two real codes stays fairly close to the data even for the plain AE, so the two rows look similar; the AE's failure shows in random sampling (s15), not here.
Part 6: Limitations, extensions, comparison, wrap-up
17. Blur: original vs VAE reconstruction; a demo averaging two sharp cats to show why MSE gives blur.
18. Extensions: beta-VAE (points back to slide 13), VQ-VAE (codebook illustration), and Stable Diffusion as an animated diagram (512x512x3 image -> VAE encoder -> 64x64x4 latent -> diffusion steps -> VAE decoder -> image), with the 48x compression number. No live Stable Diffusion model.
19. VAE vs GAN vs diffusion: comparison table, a clickable trilemma triangle (quality, variety, speed), and "Gary", a mode-collapsed GAN that only draws one cat.
20. Wrap-up: three rows: plain encoder-decoder, autoencoder, VAE, and what each solves.
21. Questions: mascot with a question mark, plus credits.

## Design
- Dark navy background (#14112B) with bright accents: coral #FF6B6B, sunflower #FFD166, mint #06D6A0, sky #4CC9F0, lilac #B388FF. One accent per part, in that order; Part 6 comes back to coral.
- Fonts (vendored with @fontsource): Fredoka for headings, Inter for body, JetBrains Mono for numbers and code.
- Large type for a projector: body at least 24px, headings 56px+ (scaled down on phones). High contrast; no thin grey text.
- A simple original cat mascot drawn in SVG (not a known character), reused with different expressions.
- Humour: at most one joke per slide, never at anyone's expense, never in place of a technical claim. Running gags: Photocopier Cat, here be monsters, Gary the GAN.
- Motion: short and purposeful (200-400 ms); respect prefers-reduced-motion.

## Navigation
- CSS scroll-snap, one section per screen. PageDown/PageUp (clickers), arrow keys and space move between sections. Sliders must not swallow navigation keys after you let go of them.
- Touch: swipe between sections, plus small on-screen prev/next buttons that fade out when idle.
- F toggles fullscreen. ? shows a small keyboard help overlay. URL hash (#s05) jumps to a slide.
- A thin progress bar and part label at the top.
- First visit: a short loading screen with a progress bar while models download, then every model is warmed up so no demo pauses the first time it is used.
