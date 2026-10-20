# Autoencoders & VAEs: Presentation Script

Tuesday 20 October 2026, 18:30 · @Danish · MAAI901 Deep Learning for Visual Data · deck: autoencoders.danishali.dev

Expect about 28 minutes of speaking, or about 34 minutes with slide changes, demos and pauses, before questions. Lines in italics are slide cues, not things to say. Notes in brackets are short explanations you can say out loud or skip. **Analogy** marks an analogy; each slide has at most one main one, and most slides show a short version of it in the top-right corner. The slide ids (#s07, #s06b) match the URL hash, so you can jump with them.

| Part | Approx. words | Approx. time |
| --- | --- | --- |
| 1. From classification to generation | 298 | 2.5 min |
| 2. Autoencoders (incl. undercomplete, overcomplete and the other types) | 1,319 | 10 min |
| 3. Why autoencoders can't generate | 180 | 1.5 min |
| 4. Variational autoencoders | 1,168 | 9 min |
| 5. Limitations, alternatives (GAN), comparison, variants, evaluation, wrap-up | 670 | 5 min |
| Total | \~3,635 | \~28 min |

Times assume a steady presenting pace of about 130 words per minute. If you need to cut, these are safest, in order: the Variants slide (keep only Stable Diffusion), "Making overcomplete work" (one tab, name the rest), PCA vs autoencoder (say "nonlinear PCA" and move on), the two "Watch it learn" replays (start them and move on), and the "if asked" ELBO line. Keep "Every face is a point", the GAN slide, the comparison and the evaluation slide: the first is the clearest picture of a latent space, and the rest are what a master's audience will ask about.

### The story in one breath (say this to yourself before you start)

Classifiers can tell cats apart but can't draw one → squeeze a cat through a bottleneck and you get features, without labels → but the code space has holes, so you can't sample it → a VAE encodes a cloud, not a point, and the KL term packs the clouds into one round blob → the reparameterization trick makes that trainable → now you can sample new cats, blurry but honest → and that VAE still sits inside Stable Diffusion today.

## Part 1: From classification to generation

*Slide #s01: title.*

- Hi, I'm Danish. Over the last 4 weeks we've been going over different discriminative models like CNNs that can classify images into categories or help you find information about said images. Today we're going to go over generative AI. Our first step into it will be covered over this presentation.
- Autoencoders and Variational Autoencoders. Teaching a network to imagine cats.
- I hope there's at least a few cat people here because this presentation is pretty cat heavy.
- Every model we see today saw about 29,800 cat faces.
- For the dog people… there are a few dogs on the next slide.
- Everything runs live in this browser: trained in PyTorch, exported to ONNX. The page itself is a small deployment demo: no server, nothing leaves the laptop.
- → "Let's see what the networks we know do with a cat."

*Slide #s03: a CNN classifier taking an image and outputting "cat". Flip to Generative, then back, then "ask the classifier to draw a cat".*

- So what is the difference between discriminative models like CNNs and generative models like VAEs?
- Well, discriminative models learn basically: given an image, what is it? They can assign labels to images.
- *(Click a thumbnail.)* It tells things apart.
- Generative models on the other hand do not need labels to learn. They learn the distribution of the images themselves and therefore can produce new ones.
- *(Flip to Generative.)* p(image): you can produce new ones. This cat is from our VAE (we'll get into the details of it later).
- **Analogy:** think of it like a judge at a pet show figuring out which animals are cats vs dogs.
- While the generative models are painters on the sidelines who can draw new cats after seeing the existing pets at the show.
- *(Flip back, "ask the classifier to draw a cat".)* The judge obviously can NOT paint a cat though. As you can see…
- → "So how do generative models do this?"

## Part 2: Autoencoders

*Slide #s06: a short history. 1986 Rumelhart, Hinton & Williams; 2006 Hinton & Salakhutdinov; 2008 Vincent et al. on denoising.*

- Autoencoders transform complex data into a simpler, more meaningful representation called the latent space.
- They've existed for almost 40 years, with the first paper published in 1986.
- 1986 Rumelhart, Hinton & Williams; 2006 Hinton & Salakhutdinov (Science); 2008 Vincent et al. (denoising).
- Even newer generative models like Stable Diffusion have a VAE inside them, and the first version of DALL-E used a discrete VAE as well.
- → "Before cats, let's do this with people."

*Slide #s06b: "Every face is a point". An illustration, inspired by Deepia's VAE video: hand-drawn faces, not a trained model. Each press builds one step: one face and its two numbers; a crowd of faces on the chart; a walk along one axis; then the axes lose their names. Drag on the chart to redraw the face.*

- Autoencoders are types of neural networks that learn how to efficiently represent data.
- This illustration is inspired by a video from Deepia, but I think it explains latent space really well.
- You can look at any person and use so many classifications to describe who they are: their age, gender, hair colour, height and so on.
- This chart is a latent space where we use just the smile and the hair length to describe a person.
- You can see that I can move along the axis to change the characteristics of the person.
- Autoencoders work on the same principle of finding the fewest features we can use to describe something. It probably isn't the best representation, but it is a very compressed one.
- This can then help the autoencoder to learn the features that make up that object (it can be a person, or in our case a cat) so it can rebuild one.
- → "So what does an autoencoder look like?"

*Slide #s04: the network diagram. Flipping the switch plays the layers from the input out to the ending: a classifier head, or the bottleneck z (4,096 → z · 256) and the transposed convs. Hover a layer for its real shape.*

- *(Flip classifier head ↔ decoder; hover a layer.)* 3×64×64 → … → 256×4×4 = 4,096.
- Autoencoders basically have three parts: an encoder, which is very similar to the neural networks we've been looking at; a squished-down area called the bottleneck (written as z); and a decoder, which uses flipped, or transposed, convolutions to grow the image back.
- → "So what actually goes in, and how small should z be?"

*Slide #s07: "Undercomplete: a cat is 12,288 numbers". Each press builds one more step: the cat; its pixel grid with one pixel's RGB; pixels flying into a column of 12,288 numbers; the encoder, drawn as its four real conv layers (32×32×32 → 64×16×16 → 128×8×8 → 256×4×4), squeezing them into two neurons with live values; the two neurons turning into the two axes, with the cat landing as a dot; then 600 training cats on the same axes.*

- We've got 64×64 pixels, and each has RGB values, so a total of 12,288 numbers.
- That's our input. So in an undercomplete autoencoder we can have, for example, 4 real convolution layers that each halve the picture and add channels, all the way down to two neurons.
- These are called latent dimensions. So 2 neurons would be 2 dimensions.
- Since z, 2 numbers, is smaller than the input, 12,288, this is called undercomplete.
- **Analogy:** it's like packing two weeks of clothes in a carry-on. It can't take everything. It has to only keep what matters.
- We can obviously add more dimensions to add more features and get better results, but we'll get into why that can be an issue.
- As you can see here though, each cat can now be represented in a graph showing the latent space.

*Slide #s08: "Overcomplete: room to copy". A tapered hourglass whose neck follows the slider: 2, 16, 64, 256 (undercomplete, live, with the summed squared error for the picked cat), then none (overcomplete: loss 0.000, Photocopier Cat, "features learned: none", "it's just copying"; an illustration, since the output is the input).*

- If you add more and more dimensions, such that the number of dimensions matches your input (or even exceeds it), our model basically becomes useless.
- Because it hasn't learned anything. The easiest thing to do in this case would just be to copy the input and output it exactly. A photocopier cat, if you will.
- **Analogy:** bringing the whole textbook into the exam.
- The loss keeps getting better and at the end it's perfect. But that perfect score is just a lie.
- This means it also doesn't know what features make up the cat.

*Slide #s14: "Making overcomplete work: add a rule". Four tabs. Same job, one extra rule about z each: sparse, denoising, contractive, convolutional.*

- So how do we make overcomplete autoencoders work? We add rules.
- We can do this in a few different ways.
- Sparse means only a few neurons may fire at once.
- Denoising means we add noise to the input, so it isn't possible to just… copy the input.
- Contractive means a small change to the input barely moves z.
- And convolutional just means using conv layers, which is what we've been using the whole time.
- These rules are what let you use big, overcomplete codes. Our demo models keep z smaller so they run in the browser.

*Slide #s10: input, output, the squared-error heatmap, and the MSE and BCE formulas. Each symbol explains itself on hover or tap.*

- So how do we train this?
- We compare what comes out with what went in. That difference is the reconstruction loss: how badly the model rebuilt the cat. We saw this loss number previously when looking at overcomplete codes.
- The heatmap shows the error pixel by pixel. Bright means wrong. It is usually the eyes or nose, because fine details can get lost in the bottleneck.
- The standard choice is MSE (mean squared error): for every pixel, the difference is squared and then averaged.
- There's also BCE (binary cross-entropy), which treats each pixel as a probability between 0 and 1. Same job, just a different way of measuring.
- Then we just go back to normal training: backpropagation pushing the weights to make this number smaller.
- But do we notice something? No labels. The input IS the target. **Analogy:** it's like a test where the question sheet is also the answer key. That's called self-supervised learning.
- This matters because unlabelled images are everywhere and labelling them is expensive.
- → "Let's watch this loss do its job."

*Slide #s11: "Watch it learn". A replay of ae_2's training, snapshotted at log-spaced training steps (0, 8, 16, 32 … 6,870): the loss per step on a log axis, a slider with "replay training", four cats as the model rebuilt them at that point, and the latent map at that point. Arriving with the keys going forward starts the replay.*

- This is a replay of our real 2-number autoencoder being trained.
- At the start every cat comes out as a grey mush and all the codes sit at one spot.
- Each step the network asks which small nudge to its weights brings the loss down the fastest, and takes it. And we repeat.
- Watch the cats and the map together. The cats get their colours and faces back, and the map spreads out and organises itself: dark cats drift towards one side and light cats to the other.
- Nobody told it about fur colour. With only two numbers, it decided that was the most useful thing to remember. That is its version of smile and hair length from our faces chart.
- → "So that's how an autoencoder learns. You might have seen something like this before: PCA."

*Slide #s12: PCA versus autoencoder, k = 2, 16, 64, 256.*

- You may have met PCA, principal component analysis. It's the classic way to compress data: find the few straight directions where the data varies most, and keep only those.
- Here's the connection: an autoencoder with only linear layers, trained with MSE, learns exactly the same thing as PCA. That was shown back in 1989, by Baldi and Hornik.
- The difference is the nonlinear activations. **Analogy:** PCA is a straight ruler: it can only measure along straight lines. An autoencoder is a tape measure: it can bend to follow curves. Real cat faces don't sit on straight lines, so the tape measure fits better.
- *(Click k = 16.)* Same cats, same 16 numbers each. The number under each image is the error, so lower wins. On average the autoencoder rebuilds the cats better.
- So you can think of a deep autoencoder as nonlinear PCA. The trade-off: PCA is instant and exact, while the autoencoder has to be trained.
- → "So what are autoencoders actually good for?"

*Slide #s13: four tabs, each one live. Compression, denoising, features, anomaly detection.*

- *(Compression tab, drag the slider.)* Compression: the code is a much smaller version of the image. 12,288 numbers down to 64 is 192 times smaller. Smaller codes mean blurrier cats.
- *(Denoising tab, drag the noise slider.)* Denoising: give it a noisy or damaged image and it rebuilds a clean one, because it only knows how to draw clean cats.
- *(Features tab.)* Feature extraction: throw away the decoder and hand the code z to another model, for example a classifier. Notice the nearest cats share lighting and pose: that's what the code captured.
- *(Anomaly tab, my favourite.)* Anomaly detection: train it only on normal data. **Analogy:** it's like a security guard who only knows the regulars. Anything unfamiliar gets rebuilt badly, so a high error flags it. *(Draw on the cat or pick a not-a-cat; point at the histogram.)* This is how you'd spot defects on a factory line or odd medical scans.
- One honest blind spot: an upside-down cat still passes. Low error means cat-ish texture, not a proper cat.
- → "So it learns features, compresses, cleans up and spots oddities. But can it make a brand-new cat?"

## Part 3: Why autoencoders can't generate images

*Slide #s15: ae_2's latent map of 2,000 training cats. Click anywhere to decode that point; "random point" keeps a tally; "what lives here?" covers the map with a grid of decodes.*

- So can we generate new cats? The idea: throw away the encoder, pick a random point in the latent space, and decode it. Just like the faces chart: pick a point, draw the face.
- *(Click inside the cloud.)* Here, in seen territory, we get a cat. *(Click an empty area, then "random point" a few times.)* But random points mostly land in the gaps, and we get something that isn't really a cat.
- Why? The encoder maps every training cat to one exact point. Those points end up in clusters with big empty gaps between them, and in an irregular shape. We don't know where the good regions are, or what range to sample from.
- **Analogy:** think of an old map where only the cities are drawn and everything else says "here be monsters". The decoder has only ever visited the cities.
- *(Press "what lives here?")* This decodes the whole map at once. Near the cats you get cats. Away from them, washed-out ghosts.
- So the bottleneck solved learning features, but not generating. We need a latent space with no gaps, where any point is a cat. That's what the VAE does.

## Part 4: Variational autoencoders (VAEs)

*Slide #s16: where VAEs come from. Kingma & Welling 2013, and Rezende, Mohamed & Wierstra 2014.*

- Created by Diederik P. Kingma in his 2013 paper "Auto-Encoding Variational Bayes", with Max Welling.
- He also co-created the Adam optimizer, which we've all used sometime this semester, probably without knowing it.
- He was a co-founder of OpenAI, later worked on diffusion models at Google, and joined Anthropic in October 2024.
- There was also a 2014 paper that independently published the same idea, by Rezende, Mohamed & Wierstra at DeepMind.
- → "Before the idea itself, we need four words from probability."

*Slide #s17: "Probability in one minute". A heatmap of a joint p(x, z), its marginals on the edges, and a crosshair you can drag: the horizontal slice is p(x | z), the vertical slice is p(z | x). The notes build up press by press.*

- To explain VAEs we need a tiny bit of probability. One picture covers it.
- *(Press.)* This heatmap is the joint distribution, p of x and z: how likely an image and a code are together. Here x is squashed to one number so we can draw it.
- *(Press.)* Add up the columns and you get p of x, all the images. Add up the rows and you get p of z, all the codes. Those are the marginals.
- *(Press, drag the crosshair.)* Take one slice and rescale it, and you get a conditional. p of x given z: the images one code makes. That's the decoder. p of z given x: which codes explain this image. That's called the posterior, and it's the encoder's job.
- *(Press.)* The catch: computing the posterior exactly means adding up over every possible code. For 12,288-number cats, nobody can do that.
- → "So what does a VAE do instead?"

*Slide #s18: "Two worlds". Left: p(x), a lumpy shape made of real training-cat photos. Right: p(z) = N(0, I), a round glow. Press for the decoder arrow, then the encoder arrow. Tap anywhere in p(z) to decode that exact code; pick a cat to see its cloud q(z | x).*

- Picture two worlds. On the left, p of x: real cat photos. It's a strange, lumpy shape that nobody can write down. We only have samples, our 29,800 cats.
- On the right, p of z: the latent world. Here's the key move: we choose this one, a simple round normal distribution. That's called the prior.
- *(Press, tap in p(z).)* The decoder bridges right to left: any point here comes out as a cat over there.
- *(Press, pick a cat.)* The encoder goes the other way. We can't compute the true posterior, so we approximate it with a simple Gaussian, q of z given x. That approximation is the "variational" in variational autoencoder.
- So generating is easy: sample the round world, decode it into the strange one.
- → "But how do we train both bridges at once?"

*Slide #s20: "The VAE's loss: the ELBO". The ELBO with labelled braces (data consistency, latent-space regularisation) and loss = −ELBO; below it a Gaussian you can drag against N(0, 1), the live KL and "snap to N(0,1)"; the two terms written out at the bottom.*

- From Bayes' rule you can derive one training objective, the ELBO, the Evidence Lower Bound. It looks scary, but it's just two parts.
- Part one, data consistency: encode a real cat, decode it, and check how close it came back. That's our same MSE as before. Rebuild the cat.
- Part two, regularization: the KL divergence, a kind of distance between two distributions. It measures how far each cat's cloud is from our round prior.
- *(Drag the Gaussian, then "snap to N(0,1)".)* It's zero only at mean 0, standard deviation 1. **Analogy:** think of KL as a leash tying every cloud to the middle of the park.
- So a VAE's loss is a regularized reconstruction loss: rebuild the cat, and keep the latent space shaped like a normal distribution.
- *(If asked.)* It's a lower bound on the log-likelihood of the data, which we can't compute directly. Push the bound up and the likelihood rises with it.
- → "So how does the network actually produce these clouds?"

*Slide #s19: the vae_2d_b1 map with the picked cat's cloud drawn as a glowing Gaussian, and a pipeline strip below: encoder → μ, σ → N(μ, σ²) ~ z → decoder, with the live numbers.*

- A plain autoencoder maps each cat to one exact point. A VAE's encoder outputs two numbers per dimension instead: a mean, μ, and a spread, σ. That's a Gaussian, a little cloud.
- *(Pick a cat; point at the strip at the bottom.)* Encoder, μ and σ, sample a z from the cloud, decoder.
- *(Click "sample again".)* Same cat, a different spot in its cloud, almost the same cat back.
- **Analogy:** it's the difference between an exact street address and a neighbourhood. The decoder has to handle the whole neighbourhood, so the gaps start to fill.
- → "But there's a problem with that sampling step."

*Slide #s25: the computation graph, opening on naive sampling with the lead "Can we backprop through a random sample? No: the gradient stops at the dice." Flip to Reparameterization: z = μ + σ × ε. "Resample ε" animates the trick on the mini map: a white unit blob with ε in it is scaled by σ, then shifted by μ.*

- To train, gradients have to flow back through every step. But can we backprop through a random sample? No. **Analogy:** you can't ask a dice how it should change to roll a bit higher.
- *(Naive sampling showing.)* The gradient stops right here, at the dice. So the encoder never learns.
- *(Flip to Reparameterization.)* The trick: roll the dice first. Sample ε from a standard normal, then compute z = μ + σ·ε. Stretch it by σ, shift it by μ.
- *(Click "resample ε"; drag μ and σ.)* Same result as sampling from the cloud, but now it's just multiply and add. The gradient flows straight through μ and σ, and we can train with Adam. There's Kingma again.
- Small detail: the encoder actually outputs log σ², which keeps the numbers stable and σ always positive.
- → "Let's watch it train."

*Slide #s22: "Watch a VAE learn". A replay of vae_2d_b1's training at the same log-spaced steps as before: reconstruction and KL per step, four cats and the latent map per snapshot. Arriving with the keys going forward starts the replay.*

- *(Arrive with the clicker; the replay starts.)* Same replay as before, but for our 2-number VAE.
- At the start, every cat sits in the same spot. Then the KL pulls everything into one round blob around zero: the shape of our prior.
- The KL weight ramps up slowly at first, which is called KL warm-up. It lets the decoder learn to draw cats before the space gets squeezed. Watch the KL curve fall from the thousands.
- Meanwhile the reconstruction keeps improving: both parts of the loss, working at the same time.
- → "So what happens if we change how hard the KL pulls?"

*Slide #s21: the KL dial. β = 0, 0.1, 1, 4 on a big map, the tug of war, and "what lives here?".*

- The two parts of the loss fight each other. Reconstruction wants every cat in its own spot. KL wants everyone bunched in the middle. **Analogy:** it's a tug of war.
- *(Drag β from 0 to 4.)* At β = 0 there's no KL: scattered islands, just like the plain autoencoder. At β = 4 KL wins: everything's squashed and the cats get blurrier. β = 1 is the standard VAE.
- *(Press "what lives here?" at β = 0, then β = 1.)* Remember the monsters? At β = 0 the gaps still decode to junk. At β = 1, every point in the round region is a cat.
- → "So now we can finally generate."

*Slide #s26: 24 samples from a bigger VAE (10M weights, trained overnight) beside 24 from a bigger plain AE, naive or "fair" (its own code statistics); re-roll draws new ones from precomputed pools. A small live VAE map on the side.*

- Once it's trained, throw away the encoder. To make a new cat, sample a random point from the normal distribution and decode it. *(Re-roll.)*
- Every one of these is a new cat that doesn't exist. Soft, but cats.
- *(Point at the plain-autoencoder grid; try "fair".)* The plain autoencoder sampled the same way gives tiger-stripe texture or collages. Its codes never lived near that round blob.
- These grids come from bigger models I trained on my desktop, too big for the browser, so their samples are precomputed. *(Click the small map.)* This little map is the live VAE: anywhere you click, a cat.
- → "And because there are no gaps, we can walk between cats."

*Slide #s27: interpolation. Pick A and B; three rows of 8 frames: a pixel crossfade, the bigger plain AE's walk, the bigger VAE's walk; a scrubber with a big frame per model.*

- *(Pick A and B; drag t.)* Encode two cats, walk the straight line between their codes, and decode each step.
- Top row: just fading the pixels stacks two ghost cats. **Analogy:** it's a crossfade, not a morph.
- The plain autoencoder glitches in the middle, because the points between its codes were never trained.
- The VAE morphs smoothly, a bit softer, but every frame is a cat. That's the continuous latent space paying off.
- → "But you've noticed every VAE cat is a bit… soft."

## Part 5: Limitations, alternatives and wrap-up

*Slide #s28: blur. Loupe over original vs reconstruction, and the pixel average of 1 up to all 29,331 training cats.*

- The biggest weakness of VAEs is blur.
- Part of it is MSE. If several sharp answers are possible, MSE's best bet is their average, and an average of sharp images is blurry. **Analogy:** it's like being unsure which door a friend will come through and pointing at the wall between them.
- *(Drag the averaging slider to the end.)* Average all 29,331 training cats and you get one beige ghost.
- The other part is KL: the leash limits how much detail each code can carry.
- → "So is there a way to get sharper images?"

*Slide #s28b: "GAN as an alternative?". Left: the GAN diagram, noise → generator → fake, a real cat, both into the discriminator → real or fake?, and the "fool me better" arrow back to the generator. Right: nine big samples from Gary, the real GAN trained on our cats, with "new samples".*

- The main alternative at the time was the GAN, a Generative Adversarial Network. The next presentation covers them properly, so just the idea.
- Two networks: a generator draws images, and a discriminator tries to tell real from fake. **Analogy:** it's a forger and a detective, each getting better because of the other.
- There's no reconstruction loss and no averaging. The detective punishes anything that looks fake, including blur, so GAN images come out sharp.
- The cost: training is unstable, and GANs can mode collapse: the forger finds one painting that fools the detective and paints it forever.
- → "So how do all three compare?"

*Slide #s30: a comparison table of VAEs, GANs and diffusion, revealed row by row, the trilemma triangle, and real samples from a bigger VAE, a GAN (Gary) and a small diffusion model.*

- *(Reveal the rows.)* VAEs: stable training, they cover the whole dataset, they have an encoder, but they're blurry. GANs: sharp in one pass, but unstable and can collapse. Diffusion: sharp and varied, but slow, with many denoising steps per image.
- *(Point at the triangle.)* **Analogy:** it's fast, good, cheap: pick two. People call it the generative trilemma: quality, variety, speed. Each model gets two.
- *(Bottom row, "new samples".)* Real samples from three models I trained on the same cats: our VAE, a GAN I call Gary, and a small diffusion model.
- They're building blocks, not rivals. Stable Diffusion's VAE is trained with a GAN-style discriminator to make it sharper.
- → "People have also built a lot on top of the basic VAE."

*Slide #s29: "Variants: conditional VAE, β-VAE, VQ-VAE", and the Stable Diffusion card. "Remember this?" opens a note about the KL dial. It does not leave the slide.*

- Conditional VAE: a plain VAE can't take requests, you just get a random cat. Feed it a label too, and you can ask for a tabby.
- β-VAE: that's the KL dial we just turned. A bigger β gives a tidier, more disentangled space, where each axis means one thing, like smile and hair on our faces chart. The cost is blurrier output.
- VQ-VAE: instead of a smooth cloud, it uses a fixed dictionary of codes. **Analogy:** like paint-by-numbers. It's sharper, and a cousin of it powered the first DALL-E.
- And Stable Diffusion: its VAE squeezes a 512×512 image down 48 times. Diffusion runs in that small space, and the very last step is a VAE decoder.
- → "Two last questions for any model: how do you measure it, and should you use it?"

*Slide #s30b: three cards. "How do we measure it?" with live numbers computed in the browser on the 64 held-out cats; "Where it's used"; "Use it responsibly".*

- *(Point at the live numbers.)* PSNR: how close the rebuild is, on 64 cats the models never saw, computed live. Higher is better, and it climbs with the size of the code.
- Generation has no answer key, so instead we compare crowds: FID compares the features of real and generated images. To be honest, this deck shows reconstruction, not FID.
- Uses: anomaly detection, denoising, compression, synthetic data, and the latent space inside Stable Diffusion.
- Responsibility: a model learns its data's bias. Ours only saw front-facing cats, so a side-on cat looks "abnormal". With people, that's a fairness problem. And the same tools can make faces that never existed, so consent and deepfakes matter.
- → "So to wrap up."

*Slide #s31: the wrap-up timeline. Each row is a main point, revealed on the next press.*

- *(Press per row.)* CNNs are the judges: they tell images apart, but they can't paint.
- An autoencoder squeezes a cat through a bottleneck and learns features with no labels, but its map has holes: here be monsters.
- A VAE gives every cat a cloud instead of a point, puts every cloud on a KL leash, and the reparameterization trick makes it trainable.
- The result is a model you can sample new cats from, still at the heart of today's best generators.
- Thank you. The next presentation will show you the forger and the detective properly.

*Slide #s32: Questions. Leave it up during Q&A.*

## Appendix A: analogy cheat sheet

| Slide | Idea | Analogy |
| --- | --- | --- |
| #s03 | Discriminative vs generative | A pet-show judge can't paint a cat; the painters on the sidelines can |
| #s06b | Latent space | Describe a face with two numbers (smile, hair); everyone becomes a dot on a chart (Deepia) |
| #s07 | Undercomplete | Packing two weeks of clothes into one carry-on |
| #s08 | Overcomplete | Bringing the whole textbook into the exam: you copy, you don't learn |
| #s10 | Self-supervised | The question sheet is also the answer key |
| #s11 | Training | A ball rolling downhill; which nudge helps most (3Blue1Brown) |
| #s12 | PCA vs AE | A straight ruler vs a tape measure |
| #s13 | Anomaly detection | A security guard who only knows the regulars |
| #s15 | Holes in latent space | An old map: cities, and "here be monsters" |
| #s17 | Marginals, conditionals | A spreadsheet of grades and ages: column totals and one rescaled row |
| #s18 | Prior and decoder | A translator between cat photos and a simple invented language |
| #s20 | KL term in the ELBO | A leash tying every cloud to the middle of the park |
| #s19 | Encode a distribution | An exact address vs a neighbourhood |
| #s25 | Sampling breaks gradients; the trick | You can't ask a dice how to roll higher; so roll the dice first, then stretch by σ and move by μ |
| #s21 | β dial | A tug of war between rebuilding and packing together |
| #s27 | Interpolation | Crossfade vs morph |
| #s28 | Blur | Pointing at the wall between two doors |
| #s28b | GAN | A forger and a detective |
| #s30 | Trilemma | Fast, good, cheap: pick two |
| #s29 | VQ-VAE | Paint-by-numbers with a fixed palette |

## Appendix B: how this maps to the subject outline

- LO1 (structure of visual data, suitable approaches): an image as 12,288 numbers (#s07), discriminative vs generative (#s03), and when a reconstruction model suits a task (#s13).
- LO2 (design and implement CNNs and generative models): the conv encoder and transposed-conv decoder (#s04), the model families trained for this deck (live in the browser), plus a bigger VAE, a GAN and a small diffusion model for the comparisons.
- LO3 (pre-trained and diffusion models): the VAE inside Stable Diffusion (#s29), and how VAEs, GANs and diffusion fit together (#s30).
- LO4 (evaluate and compare with appropriate metrics): reconstruction error and PSNR on held-out cats, PCA vs AE (#s12), FID and its caveats (#s30b).
- LO5 (generative AI in production): anomaly detection, denoising, compression and synthetic data (#s13, #s30b); the deck itself is a deployed ONNX Runtime Web model (week 10).
- LO6 (communicate rigorously): cited history (#s06, #s16), honest limitations (blur, blind spots, which samples are live and which are precomputed), and responsible use (#s30b).
