# Cat VAE presentation: spec

## Goal
A 25 to 45-minute class presentation (about 34 minutes of speech in docs/script.md) (MAAI901 Deep Learning for Visual Data) on autoencoders and VAEs, built as a single scrolling web page instead of slides. Every key idea is demonstrated live with small models trained on 64x64 cat faces. The script lives in docs/script.md. After class, the same site stays public at https://autoencoders.danishali.dev for anyone to explore.

## Setting
- Presented from any device with a modern browser: a laptop on a projector, a tablet or a phone. Must look right at 1920x1080 and 1366x768, and stay usable on a phone in portrait. Phone layouts stack when there is room; a demo that would push a model image below the first screen sits side by side instead, at a smaller size.
- Speaker notes are NOT part of the site (the presenter reads them from the script doc on a phone).
- Tone: fun, colourful, a bit of humour, still appropriate for a master's class.

## Architecture
- training/ (desktop, RTX 4070 Ti, run once): PyTorch dataset loaders, models, train.py and export.py. All heavy work happens here: training, and precomputing anything slow (PCA, latent maps, classifier visualisations, statistics). Exports go to site/models and site/data.
- training/snapshots.py replays ae_2 and vae_2d_b1's training (same settings and seed, separate copies under runs/snapshots/) and records, after every epoch, 600 map cats' codes and 4 showcase reconstructions; export.py turns that into data/training.json and data/training_recon.jpg. export.py also writes data/latent_sizes.json (ae_2/16/64/256 codes of the same 600 cats, PCA to 2-D).
- Training outputs: training/runs/<name>/model.pt (run config, state_dict, per-epoch loss history) plus epoch_XXX.png previews; cnn_cls also writes metrics.json. `--quick` smoke tests write to training/runs/quick/ so they never count as finished runs for `--skip-existing`.
- site/: fully static HTML, CSS and vanilla JavaScript ES modules. No build step, no framework, no backend. onnxruntime-web (1.30, the single-file WASM bundle) is vendored into site/vendor/ort-1.30.0 (the version in the folder name, because vendor files are cached as immutable) and runs the small models in the viewer's browser. WASM only: the WebGPU builds need a 26.8 MB .wasm, over the 25 MB file limit, and WASM already runs each of our models in a few milliseconds. Multi-threaded (up to 4 threads) because the page is cross-origin isolated (COOP/COEP headers, see Hosting); single-threaded wherever it isn't. Maths is typeset with KaTeX 0.16.47, vendored as an ES module with its CSS and woff2 fonts (site/vendor/katex-0.16.47), the LaTeX look of the 3Blue1Brown and Deepia videos. Fonts are vendored woff2 files (Latin plus Greek for the maths).
- The browser only runs small encoders, decoders and the classifier on 64x64 images. Each call should take under ~50 ms on a mid-range phone; anything slower is precomputed.
- A service worker (site/sw.js) caches every file after the first visit, so the site keeps working offline (classroom Wi-Fi can fail). It precaches the shell on install; models, data and the runtime are cached as the loading screen downloads them; everything is served cache-first. The page itself is cached under "./", not index.html (hosts with clean URLs redirect /index.html to /, and a redirected response can't answer a navigation). Navigations to / and /index.html (query strings ignored) are that page. Any other navigation that is not a real cached or network file is answered with 404.html and status 404, online and offline; that response is not stored under the missing address. tools/stamp_site.py writes site/precache.json (every file and its size) and the cache VERSION in sw.js (a hash of data/files.json, models/manifest.json and the other site files), and checks the host limits. It sorts paths and writes LF the same way on every OS, and .gitattributes keeps site/ files LF in checkouts, so Windows and Linux give the same version. export.py runs it, and you run it after any other change in site/. On localhost the service worker is off unless you open the page with ?sw=on (and ?sw=off turns it off again).
- Shell (site/js): main.js (navigation, top bar, overlays, lazy slide modules), loader.js (first-visit downloads and model warm-up), offline.js, models.js (sessions, encode/decode/classify, tensors, sprites, ModelMissing and the "still training" card), ui.js (shared widgets), mascot.js. Each slide's demo lives in js/slides/sNN.js exporting init(sectionEl) and optional enter(info)/leave()/step(dir); enter gets { direction, byKey } (byKey: the presenter arrived with the keys or the nav buttons), and step(dir) can take a next/previous press inside the slide (s31 uses it to reveal rows; js/anim.js builder() gives any slide the same press-by-press build); main.js imports it when the slide comes within one screen. Elements that a finger drags inside a demo get class="interactive" so swipes starting there don't change slides. Shared demo pieces: js/viz.js (error heatmap and histogram), js/state.js (the image picked on s03, reused by s04 and s05), js/netdiagram.js (the layer diagram with real shapes, used by s04 and s05). Slide CSS lives in css/partN.css. tools/serve.py serves site/ locally with caching off. Slide modules are js/slides/sNN.js for section id sNN. There are 32 slides. Explainer helpers: js/anim.js (tween, reduced motion, press-by-press builder), js/math.js (KaTeX: [data-tex] elements, \htmlData{tip=KEY} explainable symbols, \htmlClass{recon|kl} colours), js/learnview.js (training replays), js/decodegrid.js (a grid of decodes over a latent map).
- Size budget: total download under 60 MB as transferred (gzip), every single file under 25 MB (static host limits). Today: about 58 MB transferred, 72.8 MB on disk (the 14.2 MB runtime .wasm gzips to 3.7 MB). export.py stores the biggest models' weights as float16 (cast back to float32 inside the graph, so every onnxruntime-web backend runs them) until everything fits, but only where the float16 file still matches PyTorch within 1e-2.
- Exports (training/export.py): every file except manifest.json and files.json carries a 6-character content hash in its name (vae_128.decoder.0fa1da.onnx, showcase.72a00f.png) so hosts can cache it forever. site/models/<name>.encoder.<hash>.onnx and .decoder.<hash>.onnx (cnn_cls.<hash>.onnx for the classifier), opset 17, dynamic batch; site/models/manifest.json; site/data/ holds the showcase sprite (the 64 held-out demo cats), animals, cls_dream, PCA reconstructions, latent maps (thumbnails as JPEG: 1 MB instead of 4.7 MB as PNG), anomaly statistics, not-cats, sparse codes, and files.json listing every file as {name (logical, e.g. data/showcase.png), path (hashed), bytes}. The JSON files refer to each other by logical name; models.js (loadJSON, loadSprite) resolves logical data/ names through files.json, and manifest.json holds the hashed model file names.
- site/models/manifest.json lists every model, its latent size and files. Any demo whose model is missing shows a friendly "this cat is still training" card instead of breaking.

## Hosting
- Vercel (team "Danish's projects", where danishali.dev's DNS lives), serving site/ as is: vercel.json at the repo root sets framework none, outputDirectory site, no build. Custom domain autoencoders.danishali.dev. .vercelignore keeps training/, tools/, docs/ and design-mockups/ out of CLI uploads. design-mockups/ is not inside site/, so it is neither published nor written into the service-worker cache. Steps in docs/DEPLOY.md.
- vercel.json headers, exactly one Cache-Control rule per path (regex lookaheads keep the two index files out of the folder rules): models, data and vendor are `public, max-age=31536000, immutable`; css and js keep plain names (no build step to rewrite imports) and get `max-age=3600`; /, index.html, sw.js, precache.json, models/manifest.json and data/files.json are `no-cache`. Site-wide COOP same-origin + COEP require-corp (cross-origin isolation for threaded WASM), so the page must never load anything from another origin. .wasm as application/wasm. tools/serve.py sends the same COOP/COEP locally.
- Limits: every file under 25 MiB, at most 15,000 files (checked by stamp_site.py).
- site/404.html (the confused mascot, "here be monsters"; absolute paths because it is served at any depth). It is precached with the shell so the service worker can show it for missing addresses, including offline. Link previews: og/twitter tags in index.html, image site/img/og.jpg (1200x630 capture of the title slide, under 300 KB for chat apps). og.jpg is not precached.

## Datasets
- Cats: 29,800 cat faces, 64x64 RGB, from https://github.com/fferlito/Cat-faces-dataset (three .tar.gz parts, about 90 MB each, from raw.githubusercontent.com). Hold out 512 fixed cats (seed 0) that are never trained on; the site uses them as demo inputs.
- Classifier only: AFHQ (Animal Faces-HQ: cat, dog, wild) resized to 64x64, downloaded on the desktop from Hugging Face (e.g. huggan/AFHQv2). If unavailable, use Oxford-IIIT Pet head crops (cat vs dog). The classifier's cat class mixes AFHQ cats with fferlito cats, so it cannot learn "which dataset" instead of "which animal".

## Models
All share one conv trunk: conv 32-64-128-256 (stride 2, BatchNorm, LeakyReLU) down to 256x4x4, flattened to 4096. Autoencoders add a linear layer to the latent and a mirrored decoder (transposed convs, sigmoid). The classifier adds a linear head instead of a decoder. Teaching point: classifier = encoder + label head; autoencoder = encoder + decoder. Reconstruction loss = summed squared error per image.
| name | type | latent | used for |
| cnn_cls | CNN classifier, 3 classes (cat, dog, wild) | - | live classification (s03, s04), "ask the classifier to draw a cat" (s03). It is not used to score generated cats: it calls every decoder output a cat. |
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
1. Title. No cats on this slide. Tuesday 20 October 2026. The title is uppercase but keeps the plural s small ("VAEs", not "VAES").
2. Warm-up: "Cat people or dog people?" Show of hands. No tally.
3. Discriminative vs generative. Live cnn_cls, flip to a vae_128 cat, and the precomputed dream image.
4. Encoder and decoder. Flipping classifier-head / decoder plays the layers from the input out to that ending. No "next layer" button. Hover a layer for its real shape.
5. Identity trap. Toggle makes every layer as wide as the input. Photocopier Cat.
Part 2: Autoencoders
6. History. Rumelhart, Hinton & Williams 1986; Hinton & Salakhutdinov 2006; Vincent, Larochelle, Bengio & Manzagol 2008 (denoising). Dates kept conservative.
7. "A cat is 12,288 numbers" (js/slides/s07.js, an SVG build, one press per step): the two-numbers analogy; the cat's pixel grid with one pixel's RGB; pixels flying into a column of 12,288 numbers (3Blue1Brown, chapter 1); the encoder squeezing them into two neurons with ae_2's live values; the two neurons becoming the two axes and the cat landing as a dot (Deepia); 600 training cats on the same axes = the latent space.
8. Bottleneck. Live ae_2/16/64/256. The diagram is a tapered hourglass whose neck follows the slider.
9. "More numbers, more room": the same 600 cats in ae_2/16/64/256's latent spaces (data/latent_sizes.json; codes wider than 2 flattened to their 2 main PCA axes, with the share of the spread kept) beside the picked cat rebuilt live by each. Honest takeaway: bigger codes rebuild sharper, but a flat picture of a 16-D space keeps only about a third of it.
10. Reconstruction loss. Live MSE and BCE. Each symbol in the formulas explains itself on hover or tap. "labels used: 0 / human annotators harmed: 0" is the self-supervised joke.
11. "Watch it learn" (js/learnview.js): a replay of ae_2's training from training/snapshots.py (data/training.json, data/training_recon.jpg). Loss curve (log scale), epoch slider and "replay training", four cats as rebuilt at that epoch, and the latent map at that epoch. Arriving forward with the keys starts the replay.
12. PCA vs autoencoder. Same claim as before: linear + MSE matches the PCA subspace.
13. What they are good for. Four tabs, all live: compression (ae_2/16/64/256), denoising (dae_128 and a noise slider), features (ae_2 code, crosshair, and the photos of the four training cats with the nearest codes, from map_thumbs; decoding them through ae_2 gave four identical blurs), anomaly (ae_64, unchanged rules).
Part 3: Other types of autoencoders
14. Four tabs with a one-line lead: same job, one extra rule about z. Sparse, denoising (dae_128 vs ae_128), contractive, convolutional.
Part 4: Why autoencoders can't generate
15. ae_2 latent map. Click to decode. Seen territory vs here be monsters. On load it decodes the median training code, so the output square never starts empty. "What lives here?" covers the map with a 12x12 grid of ae_2 decodes (js/decodegrid.js): away from the cats only washed-out ghosts.
Part 5: Variational autoencoders
16. History. Kingma & Welling 2013 (ICLR 2014) and Rezende, Mohamed & Wierstra 2014 (ICML), as independent papers.
17. "Probability in one minute" (Deepia's VAE video): a toy joint p(x, z) heatmap, its marginals on the edges, and a draggable crosshair whose slices are p(x | z) and p(z | x). Notes build up to why the posterior is intractable and what "variational" means (approximate it with a Gaussian q(z | x)).
18. "Two worlds": p(x) as a lumpy shape of real training-cat photos, p(z) = N(0, I) as a round glow (the prior we choose). Decoder arrow: tap anywhere in p(z) and vae_2d_b1 decodes that exact z into a cat that lands in p(x). Encoder arrow: pick a cat and its q(z | x) cloud appears in p(z).
19. Core idea. vae_2d_b1 cloud, sample again. The cloud is drawn as a glowing Gaussian with faint 1 and 2 sigma rings; a pipeline strip below (encoder → μ, σ → N(μ, σ²) ~ z → decoder) shows the live numbers (replaces the caption).
20. Loss = reconstruction + KL. The formula bar stays on screen at short laptop heights and at 110–125% zoom. The loss is typeset with KaTeX and labelled braces (rebuild the cat / keep the clouds near N(0,1)).
21. KL dial. Beta 0, 0.1, 1, 4. ELBO callout. The ELBO callout is typeset with braces: data consistency (our L2) and latent-space regularisation. "What lives here?" decodes a grid with the selected beta's model: junk in the gaps at beta 0, cats everywhere at beta 1.
22. "Watch a VAE learn": the same replay for vae_2d_b1, with reconstruction and KL curves (each on its own log scale; KL starts in the thousands during warm-up). The map gets pulled into one round blob.
23. Backpropagation in a CNN, in three lines: forward, loss, chain rule. A plain autoencoder is differentiable.
24. Why z ~ N(μ, σ) blocks the gradient. The sample is the break, not the encoder.
25. Reparameterization. Naive vs trick, live μ/σ sliders and a decoded cat. "Resample ε" animates the trick on the mini map: a white unit blob with ε, scaled by σ, then shifted by μ (μ starts at (1.2, 0.9) so the shift shows).
26. Generating. vae_128 vs ae_128, naive or fair. Small vae_2d_b1 map.
27. Interpolation. AE vs VAE. The AE's failure was random sampling (s26), not this walk.
Part 6
28. Blur. Loupe and the pixel average.
29. Extensions. "Remember this?" opens an overlay about the KL dial. It does not navigate away. VQ-VAE is still an illustration; its card says a close cousin (dVAE) powered the first DALL-E (DALL-E 1 used a dVAE, not a VQ-VAE). Stable Diffusion is still a diagram, plus a real vae_128 round trip. The β-VAE card adds one line on the conditional VAE (a plain VAE can't take requests).
30. VAE vs GAN vs diffusion. The triangle always labels who is strong on quality, variety and speed, at 19px or more on a 1366 screen; on phones it drops below the table. Gary is labelled as a VAE illustration of mode collapse, not a GAN photograph. This repo has no GAN checkpoint.
31. Wrap-up timeline, five main rows in date order: autoencoder (1986, deep 2006, denoising 2008), CNN (1989/1998, AlexNet 2012), VAE (2013–2014), GAN (Goodfellow et al. 2014), diffusion (Sohl-Dickstein et al. 2015, Ho et al. 2020). Keys reveal one row at a time.
32. Questions. Thinking mascot, vae_128 cats, credits.

## Design
- Pixel-training chrome on warm paper (#EFE6D2) with ink (#24170F). Accents are dark stamps so labels stay readable: coral #9D2F28, sunflower #94500F (4.5:1 or better on all three paper shades), mint #1F5C3A, sky #1E3F66, lilac #6B3050. One accent per part, in that order; Part 6 comes back to coral. Frames are square, with a hard offset shadow. The orange stamp colour (#E07A3D) is the heading shadow only.
- Every cat the demos show is still a trained-model output or a dataset image (ONNX encoders and decoders, the showcase sprite, precomputed reconstructions). The pixel look is frames, type, colours and the SVG mascot. It does not replace those images.
- Fonts: Noto Sans Display Bold for headings (vendored TTF, OFL), Inter for body, JetBrains Mono for numbers and code. Fredoka stays loaded for the mascot's short labels.
- Large type for a projector: body at least 24px, headings 56px+ (scaled down on phones). High contrast; no thin grey text.
- A simple original cat mascot drawn in SVG (not a known character), reused with different expressions.
- Humour: at most one joke per slide, never at anyone's expense, never in place of a technical claim. Running gags: Photocopier Cat, here be monsters, Gary the GAN.
- Motion: short and purposeful (200-400 ms); respect prefers-reduced-motion.

## Navigation
- CSS scroll-snap, one section per screen. The snap pins the top of the section, so every demo image has to sit inside the viewport (100dvh); a taller section hides the rest. PageDown/PageUp (clickers), arrow keys and space move between sections. Sliders must not swallow navigation keys after you let go of them.
- Touch: swipe between sections, plus small on-screen prev/next buttons that fade out when idle. Only a mouse, a finger or a wheel wakes them; keys and clickers don't, so while presenting they never sit on a slide's bottom caption.
- F toggles fullscreen. ? shows a small keyboard help overlay. URL hash (#s05) jumps to a slide.
- A thin progress bar and part label at the top.
- First visit: a short loading screen with a progress bar while models download, then every model is warmed up so no demo pauses the first time it is used.
