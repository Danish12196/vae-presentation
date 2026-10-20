# Autoencoders & VAEs: Presentation Script

Tuesday 20 October 2026, 18:30 · @Danish · MAAI901 Deep Learning for Visual Data · deck: autoencoders.danishali.dev

Expect about 38 minutes of speaking, or about 44 minutes with slide changes, demos and pauses, before questions. Lines in italics are slide cues, not things to say. Notes in brackets are short explanations you can say out loud or skip. **Analogy** marks an analogy; each slide has at most one main one, and most slides show a short version of it in the top-right corner. The slide ids (#s07, #s06b) match the URL hash, so you can jump with them.

| Part | Approx. words | Approx. time |
| --- | --- | --- |
| 1. From classification to generation | 490 | 4 min |
| 2. Autoencoders | 1,290 | 10 min |
| 3. Other types of autoencoders | 220 | 1.5 min |
| 4. Why autoencoders can't generate | 180 | 1.5 min |
| 5. Variational autoencoders | 1,680 | 13 min |
| 6. Limitations, extensions, comparison, evaluation, wrap-up | 1,090 | 8.5 min |
| Total | \~4,950 | \~38 min |

Times assume a steady presenting pace of about 130 words per minute. If you need to cut, these are safest, in order: the Extensions slide (keep only Stable Diffusion), Part 3 (one tab, name the rest), "More numbers, more room", the two "Watch it learn" replays (start them and move on), and the "for the maths fans" ELBO lines. Keep "Every face is a point", the comparison and the evaluation slide: the first is the clearest picture of a latent space, and the last two are what a master's audience will ask about.

### The story in one breath (say this to yourself before you start)

Classifiers can tell cats apart but can't draw one → squeeze a cat through a bottleneck and you get features, without labels → but the code space has holes, so you can't sample it → a VAE encodes a cloud, not a point, and the KL term packs the clouds into one round blob → the reparameterization trick makes that trainable → now you can sample new cats, blurry but honest → and that VAE still sits inside Stable Diffusion today.

## Part 1: From classification to generation

*Slide #s01: title. No cats on this one.*

- Hi, I'm Danish. Today: how a network learns to imagine cats.
- Everything you'll see runs live in this browser. I trained the models in PyTorch on my desktop and exported them to ONNX, so this page is also a small deployment demo: no server, no GPU, nothing leaves the laptop.

*Slide #s02: warm-up. "Cat people or dog people?" Ask the room to raise hands. Count them yourself; the slide does not keep a tally.*

- Cat people, good news: every model today was trained on about 29,800 cat faces. Dog people, there's one dog on the next slide. Be patient.

*Slide #s03: a CNN classifier taking an image and outputting "cat". Flip to Generative, then back, then "ask the classifier to draw a cat".*

- So far in this course, almost everything we've looked at has been about understanding images.
- We've looked at CNNs, ResNet, MobileNet, Vision Transformers, CvT and a bunch of other transformers. (CvT stands for Convolutional vision Transformer, which mixes convolutions into the transformer.)
- All of these take in an image and output something about it, like a class label. These are called discriminative models. (A discriminative model learns the probability of a label given an image, p of label given image. In other words, it learns to tell things apart.)
- But what if we want to go the other way and actually create images?
- That's what generative models do. (A generative model learns something about the distribution of the images themselves, p of image, so it can produce new ones.)
- **Analogy:** a discriminative model is a judge at a cat show. It can tell you which one is the cat. A generative model is the artist who can paint a new cat. Being a great judge doesn't make you an artist. *(Click "ask the classifier to draw a cat".)* Here's the judge trying to paint.
- Today we're going to look at one of the first and most important families of generative models: autoencoders and variational autoencoders.

*Slide #s04: an encoder-decoder diagram. Flipping the switch plays the layers from the input out to the ending. Hover a layer for its real shape.*

- Could we just use the networks we already have? A classification CNN outputs class scores, not an image.
- But we can keep the convolution layers and swap the ending for a second half that grows the image back up. The first half is the encoder, the second half is the decoder.
- To grow the image back up, the decoder uses transposed convolutions or upsampling layers. (A transposed convolution is basically a convolution in reverse. Instead of shrinking the image, it makes it bigger.)
- So now we have a network that takes an image in and gives an image out. Same encoder, different ending.

*Slide #s05: the same diagram, with the toggle that makes every layer as wide as the input.*

- But there's a trap.
- If every layer in this network is as big as the input or bigger, the network can just learn to copy the input straight to the output.
- This is called learning the identity function. (The identity function is just a function where the output equals the input.)
- **Analogy:** meet Photocopier Cat. If you're allowed to bring the whole textbook into the exam, you don't need to understand anything. You just copy. The loss is perfect, and nothing was learned.
- So how do we tackle this? We need to force the model to learn features.

## Part 2: Autoencoders

*Slide #s06: a short history. 1986 Rumelhart, Hinton & Williams; 2006 Hinton & Salakhutdinov; 2008 Vincent et al. on denoising. Skim it.*

- Quick history: autoencoders go back to Rumelhart, Hinton and Williams in 1986. The deep version used on images is Hinton and Salakhutdinov, Science, 2006, and denoising autoencoders are Vincent and colleagues, 2008.

*Slide #s06b: "Every face is a point". An illustration, inspired by Deepia's VAE video: hand-drawn faces, not a trained model. Each press builds one step: one face and its two numbers; a crowd of faces on the chart; a walk along one axis; then the axes lose their names. Drag on the chart to redraw the face.*

- Before any cats, let's do this with people.
- Imagine I had to describe your face to a friend using only two numbers. I might pick how much you're smiling, and how long your hair is.
- *(Press.)* So this face is smile 0.8, hair 0.3. Two numbers. You lose a lot of detail, but your friend would still pick you out of a crowd. Probably.
- *(Press.)* Now do that for everyone in the room. Every face becomes a dot on this chart. Smiley people on the right, long hair at the top. Similar faces end up close together. That chart is a latent space. (Latent just means hidden: the features behind the picture.)
- *(Press, then drag along the bottom.)* And here's the generative part. Pick any point on the chart and you can draw the face that belongs there. Walk along one axis and only one thing changes: the smile grows, the hair stays. Going from two numbers back to a face is exactly what a decoder does.
- *(Press.)* Now the twist. I chose "smile" and "hair". A neural network doesn't get to hear those words. Nobody labels the axes, so it invents its own, and they're often a mix of things, like "lighting plus head tilt". Our job today is to make a network find its own chart, for cats.

*Slide #s07: "A cat is 12,288 numbers". Each press builds one more step: the cat; its pixel grid with one pixel's RGB; pixels flying into a column of 12,288 numbers; the encoder, drawn as its four real conv layers (32×32×32 → 64×16×16 → 128×8×8 → 256×4×4), squeezing them into two neurons with live values; the two neurons turning into the two axes, with the cat landing as a dot; then 600 training cats on the same axes.*

- So what does the network actually see?
- To the network, this cat is a grid of 64 by 64 pixels, and every pixel is three numbers: red, green and blue.
- Unroll them all into one long column and you get 12,288 numbers. That column is the input layer. (This is the 3Blue1Brown picture: a 28 by 28 digit becomes 784 numbers. Ours is just a bigger, furrier version.)
- The encoder squeezes that column down, one conv layer at a time: each layer halves the picture and adds channels. Here it squeezes all the way to two neurons, and these are the two numbers it chose for this cat.
- Now the trick from the faces slide: treat those two activations as coordinates. Neuron one is the x axis, neuron two is the y axis, and the cat becomes a single point.
- Do that for every training cat and you get this cloud of points. That's the cats' version of the face chart, except the network picked the axes itself.

*Slide #s08: the bottleneck. A tapered hourglass whose neck follows the slider: 2, 16, 64, 256, none.*

- Autoencoders force the model to learn features by making the middle narrow. This narrow part is called the bottleneck.
- The bottleneck is smaller than the input and the output, so the model can't just copy every pixel through. There isn't enough room.
- **Analogy:** it's packing for a two-week trip with one carry-on. You can't take everything, so you're forced to decide what really matters. The bottleneck is the carry-on.
- The small set of numbers it keeps is called the latent vector, z, and the space it lives in is the latent space.
- *(Drag the slider.)* 12,288 numbers down to 2 is 6,144 times smaller, and you get a ghost cat. More room, sharper cat. *(Slide to "none".)* And with no bottleneck at all, Photocopier Cat is back.
- Because the bottleneck is smaller than the input, this is called an undercomplete autoencoder: the encoder compresses the image to z, and the decoder rebuilds it from z alone.

*Slide #s09: "More numbers, more room". Four maps of the same 600 training cats: ae_2, ae_16, ae_64 and ae_256, each beside the picked cat rebuilt by that model. Codes wider than two are flattened to their two main PCA axes. Each press reveals one more model.*

- How many numbers should the bottleneck have? That's the latent dimension, one of the most important design choices. More numbers, sharper cats.
- But look at the maps. Past 2 numbers we can't draw the space anymore, so these are flattened with PCA, and that flat picture keeps only about a third of the spread.
- **Analogy:** it's like the shadow of a 3-D object on a wall. The shadow is real, but you can't tell a coffee mug from a donut from one angle. A flat picture of a 16-D space is a shadow.
- So high-dimensional latent spaces work, we just can't see them. That's why today's maps use 2-D models.

*Slide #s10: input, output, the squared-error heatmap, and the MSE and BCE formulas. Each symbol explains itself on hover or tap.*

- Now how do we train this? We compare the output to the input: the reconstruction loss.
- The most common choice is MSE. (MSE stands for Mean Squared Error. It's the average of the squared differences between each output pixel and the matching input pixel.) The heatmap shows where the error is: usually the edges and the eyes.
- Another option is BCE, which works well when pixel values are between 0 and 1. (BCE stands for Binary Cross-Entropy. It treats each pixel as a probability.)
- Then we train with backpropagation, like any other network. And notice: we don't need any labels. The input is the target.
- **Analogy:** it's a test where the question sheet is also the answer key. You just have to reproduce it through a very small memory.
- That makes this self-supervised learning. (Self-supervised means the training signal comes from the data itself, not from human labels.) Labels used: zero. Human annotators harmed: zero.
- That's a big deal, because unlabelled images are everywhere and labelled ones are expensive.

*Slide #s11: "Watch it learn". A replay of ae_2's training, snapshotted at log-spaced training steps (0, 8, 16, 32 … 6,870): the loss per step on a log axis, a slider with "replay training", four cats as the model rebuilt them at that point, and the latent map at that point. Arriving with the keys going forward starts the replay.*

- Here's what training actually looks like. This is a replay of the real run of our 2-number autoencoder: 6,870 small steps, 30 passes over the cats.
- The axis is logarithmic, because most of the visible learning happens in the first few hundred steps.
- Before training, every cat is the same grey mush and all the codes sit on one spot.
- **Analogy:** it's a 3Blue1Brown idea: training is a ball rolling downhill on the loss. Each step, the network asks "which small nudge to my weights makes the loss go down fastest?", takes it, and asks again.
- Within a few hundred steps the cats get their colours and their faces, and the map spreads out and organises itself: dark cats drift to one side, light cats to the other. After that the loss keeps creeping down, but with only two numbers the cats stay blurry.
- Nobody told it about fur. That's its version of "smile" and "hair".

*Slide #s12: PCA versus autoencoder, k = 2, 16, 64, 256.*

- What does "most important features" mean? Whatever is most useful for reconstruction under our loss. There's a nice connection to PCA. (PCA stands for Principal Component Analysis. It's a classic technique that reduces data to the few directions where it varies the most.)
- If you build an autoencoder with only linear layers and train it with MSE, it learns the same subspace as PCA. (Baldi and Hornik showed this in 1989.)
- **Analogy:** PCA is a straight ruler: it can only measure along straight lines. An autoencoder is a tape measure: it can follow curves. Cat faces live on a curved surface, so the tape measure fits better.
- *(Click k = 16.)* Same cats, same 16 numbers: the average error is lower for the autoencoder. So you can think of a deep autoencoder as a nonlinear version of PCA.

*Slide #s13: four tabs, each one live. Compression, denoising, features, anomaly detection.*

- So what are autoencoders used for? Compression: 12,288 down to 64 is 192 times smaller. Denoising: cleaning up damaged images. Feature extraction: keep the encoder, hand z to another model.
- And anomaly detection, my favourite. Train only on normal data.
- **Analogy:** it's a security guard who has only ever seen the regulars. Anyone unfamiliar gets a hard look. The autoencoder can only rebuild what it has seen, so something unusual comes back badly, and a high reconstruction error flags it.
- *(Draw on the cat, or pick a not-a-cat.)* The error lands on a histogram of normal cats. This is used for spotting defects on a factory line, or odd scans in medical imaging.
- And an honest blind spot: an upside-down cat still passes. Low error means "cat-ish texture", not "a proper cat".

## Part 3: Other types of autoencoders

*Slide #s14: four tabs. Same job, one extra rule each: sparse, denoising, contractive, convolutional.*

- The bottleneck isn't the only way to force features. These aren't four new jobs, they're four extra rules about the code.
- A sparse autoencoder can have a large hidden layer, even larger than the input. But we add a penalty that forces most of the neurons to stay switched off for any one image.
- The penalty is usually L1 on the activations, or a KL term.
- **Analogy:** it's a big orchestra where only a few players are allowed to play each song. Because so few can play, each player has to become a specialist. *(Toggle L1.)* With the penalty, about 2% of the 1,024 neurons are on. Without it, about two thirds.
- A denoising autoencoder adds noise to the input image but still asks for the clean original. It can't just copy the input anymore, because the input is corrupted, so it has to learn what a real, clean cat looks like.
- A contractive autoencoder penalizes the model if small changes in the input cause big changes in the latent vector. That makes the features more stable.
- And a convolutional autoencoder uses convolution layers in the encoder and transposed convolutions in the decoder. Plot twist: you've been watching convolutional autoencoders all along.
- And you can combine them: a convolutional denoising autoencoder is very common.

## Part 4: Why autoencoders can't generate images

*Slide #s15: ae_2's latent map of 2,000 training cats. Click anywhere to decode that point; "random point" keeps a tally; "what lives here?" covers the map with a grid of decodes.*

- So can we generate with it? Throw away the encoder, pick a random point, decode it. Remember the faces: pick a point, draw the face.
- *(Click inside the cloud.)* Seen territory: a cat. *(Click a gap, then "random point" a few times.)* But in practice, random points mostly give garbage.
- Why? The encoder maps each training image to a single point, and those points end up in clusters with big empty gaps.
- **Analogy:** it's an old map where only the cities are drawn and the rest says "here be monsters". The decoder has only ever been to the cities. Drop it in the countryside and it's lost.
- And the space is irregular: we don't know where the valid regions are, or what range to sample from.
- *(Press "what lives here?")* This is the whole map decoded at once. Away from the training cats the decoder only produces washed-out ghosts: it is guessing.
- So the bottleneck solved the feature learning problem, but it didn't solve the generation problem.
- We need the latent space to be continuous and organized, so that any point we pick decodes into something meaningful. And that's exactly what variational autoencoders do.

## Part 5: Variational autoencoders (VAEs)

*Slide #s16: where VAEs come from. Kingma & Welling 2013, and Rezende, Mohamed & Wierstra 2014. Skim it.*

- Two groups, almost at once: Kingma and Welling, December 2013, Auto-Encoding Variational Bayes; and Rezende, Mohamed and Wierstra, ICML 2014, independently.

*Slide #s17: "Probability in one minute". A heatmap of a joint p(x, z), its marginals on the edges, and a crosshair you can drag: the horizontal slice is p(x | z), the vertical slice is p(z | x). The notes build up press by press.*

- To say what a VAE does, we need four words from probability, and this one picture shows all of them. This framing is from Deepia's VAE video.
- The heatmap is the joint distribution, p of x and z: how likely an image and a code are together. Here x is squashed to one number so we can draw it.
- Add up each column and you get p of x, the distribution of images. Add up each row and you get p of z, the distribution of codes. Those are the marginals.
- **Analogy:** think of a spreadsheet of a school, with grades down the side and ages across the top. The total at the bottom of each column is a marginal. One row, rescaled to add up to one, is a conditional: "given this grade, how old are the students?"
- Take one horizontal slice and rescale it: that is p of x given z, the images one code can make. That is the decoder's job.
- Take a vertical slice instead: p of z given x, which codes explain this image. That is called the posterior, and it's the encoder's job.
- Here's the catch. To get the posterior exactly you need p of x, which means adding up over every possible code. For real cats in 12,288 dimensions, no one can compute that.
- So the VAE doesn't compute it. It approximates the posterior with a simple Gaussian, called q of z given x. Approximating a distribution you can't compute is called variational inference, and that's the "variational" in variational autoencoder.

*Slide #s18: "Two worlds". Left: p(x), a lumpy shape made of real training-cat photos. Right: p(z) = N(0, I), a round glow. Press for the decoder arrow, then the encoder arrow. Tap anywhere in p(z) to decode that exact code; pick a cat to see its cloud q(z | x).*

- So picture two worlds. On the left, p of x: real cat photos. In 12,288 dimensions that's a strange, lumpy shape that nobody can write down.
- On the right, p of z: the latent world. And here's the key assumption: we choose this one. We say the codes should follow a standard normal distribution, N of zero and I, a nice round blob. That is the prior.
- **Analogy:** it's a translator between a language nobody can write down, cat photos, and a simple one we invented, points in a round blob. Write in the simple language, read it out in cat.
- The decoder, p of x given z, is the bridge from the round world to the lumpy one. *(Tap in p(z).)* Every point I pick here comes out as a cat over there.
- The encoder, q of z given x, goes the other way: it takes a cat and gives a small cloud of codes, our approximation of the posterior.
- And that is how a VAE generates: sample the round world, which is easy, and decode it into the strange one. The encoder is only needed for training.

### The core idea

*Slide #s19: the vae_2d_b1 map with the picked cat's cloud drawn as a glowing Gaussian, and a pipeline strip below: encoder → μ, σ → N(μ, σ²) ~ z → decoder, with the live numbers.*

- Instead of a single point, the VAE's encoder outputs a distribution: a mean, mu, and a standard deviation, sigma. A Gaussian, the bell curve. We sample z from it and pass it to the decoder.
- **Analogy:** a plain autoencoder gives each cat an exact street address. A VAE gives each cat a neighbourhood. *(Click "sample again".)* Every time we pass the same cat through, we land somewhere else in its neighbourhood, but the decoder still has to rebuild the same cat.
- So the decoder learns the whole area around each point, and the gaps start to fill. But nothing yet stops the encoder from making the neighbourhoods tiny and far apart. That's the job of the loss.

### The loss function

*Slide #s20: Loss = reconstruction + KL, typeset with labelled braces. Drag the Gaussian; "snap to N(0,1)".*

- The VAE loss has two parts. The first is the reconstruction loss, the same as before. The second is the KL divergence. (KL stands for Kullback–Leibler. KL divergence measures how different one probability distribution is from another. If they're identical, it's zero.)
- It pushes each encoded distribution towards a standard normal, mean 0 and standard deviation 1. *(Drag the Gaussian away, then "snap to N(0,1)".)* It's zero only at mu 0, sigma 1.
- **Analogy:** KL is a leash. Every cat's cloud is a dog on a leash tied to the centre of the park. The dogs can wander to their favourite spots, which is reconstruction, but none can run off to a far corner, and none can shrink to a single point.
- For one dimension the closed form is one half of mu squared plus sigma squared, minus 1, minus the log of sigma squared. The training loss sums that over every latent dimension.
- The two parts have names: data consistency, rebuild the cat, and latent-space regularisation, keep every cloud close to the prior.

*Slide #s21: the KL dial. β = 0, 0.1, 1, 4; the ELBO callout; "what lives here?".*

- The two terms work against each other: reconstruction wants every cat in its own distinct spot, KL wants everyone bunched around zero.
- **Analogy:** it's a tug of war. *(Drag β from 0 to 4.)* At beta 0, reconstruction wins: scattered islands. At beta 4, KL wins: everything squashed into the middle and the cats get blurrier. Beta 1 is the balance.
- *(Press "what lives here?" at β = 0, then at β = 1.)* With no KL the gaps still decode to junk; with KL, every point in the round region is a cat.
- Formally, this combined loss is the negative of something called the ELBO. (ELBO stands for Evidence Lower Bound. It's a lower bound on the log-likelihood of the data, meaning how probable the model thinks real images are.)
- The likelihood itself is intractable, so we maximize the ELBO instead.
- For the maths fans, in one line: log p of x equals the ELBO plus the KL between q and the true posterior. That last KL can't be negative, so the ELBO sits underneath log p of x. Push the floor up and the ceiling has to go up too, and q gets closer to the true posterior on the way.
- One caveat: our reconstruction term is squared error. For a Gaussian decoder with a fixed variance, that matches the negative log-likelihood up to a constant and a scale.

*Slide #s22: "Watch a VAE learn". A replay of vae_2d_b1's training at the same log-spaced steps: reconstruction and KL per step, four cats and the latent map per snapshot.*

- Same replay, for the VAE. In the first epochs the KL weight is ramped up slowly. That's called KL warm-up: it lets the decoder learn to draw cats before the latent space gets squeezed. (Without it, the KL can win too early and the model ignores z. That failure is called posterior collapse.)
- Watch the KL curve fall from the thousands, and the map get pulled into one round blob around zero, the shape of our prior.
- The reconstruction keeps improving at the same time: the tug of war, live.

### Backprop, then the reparameterization trick

*Slide #s23: "Backpropagation, briefly". A tiny network, x → w₁ → h → w₂ → ŷ → loss, with real numbers. Each press builds one step: the forward pass; the loss; the blame flowing back, one local derivative at a time; one gradient-descent step. "Take a step" repeats it; the little valley on the right shows the ball rolling down.*

- Before the trick, a quick refresher on backpropagation, in the style of 3Blue1Brown.
- *(Press.)* Forward: the input goes through two weights and comes out as a guess. *(Press.)* Loss: one number, how wrong it was. *(Press.)* Backward: how much is each weight to blame?
- **Analogy:** the chain rule is a relay of blame. The loss tells the last layer "you were this much off". That layer keeps its share and passes the rest back, multiplied by its own local slope. Each layer only needs the message from the layer after it.
- In 3Blue1Brown's words, the gradient tells you which nudges give you the most bang for your buck: weights attached to big activations get the big nudges.
- *(Press, then "take a step" a few times.)* Then every weight takes a small step downhill, and the ball rolls down the valley. Repeat that millions of times and you've trained a network.
- A plain autoencoder is smooth from start to finish, so the blame can flow all the way back. Hold that thought.

*Slide #s24: plain AE on the left, z = encoder(x). VAE on the right, z sampled from N(μ, σ).*

- In a VAE, the only break is the sampling step. mu and sigma are ordinary numbers; you can differentiate those.
- **Analogy:** but you can't ask a dice "how should I change you to roll a little higher?" A random draw isn't a smooth function of mu and sigma, so the relay of blame stops at the dice.

*Slide #s25: the computation graph, with z = μ + σ × ε. "Resample ε" animates the trick on the mini map: a white unit blob with ε in it is scaled by σ, then shifted by μ.*

- So how do we fix this? We move the randomness outside the network.
- Sample epsilon from a standard normal, then compute z as mu plus sigma times epsilon.
- **Analogy:** roll the dice first, before the recipe starts. Then the recipe is just: take the dice result, stretch it by sigma, move it by mu. Stretching and moving are smooth, so the blame flows back through them. The dice becomes an ingredient, not a step.
- *(Toggle to Reparameterization, then "resample ε".)* On the mini map: a standard blob, scaled by sigma, then shifted by mu.
- Epsilon is just an input now, and gradients flow through mu and sigma. That's the reparameterization trick, and it's what makes VAEs trainable.
- Practical note: the encoder outputs the log of the variance, so any number is allowed and the exponential always comes back positive.

### Generating new images

*Slide #s26: 24 samples from a bigger VAE (10M weights, trained overnight) beside 24 from a bigger plain AE, naive or "fair" (its own code statistics); re-roll draws new ones from precomputed pools. A small live VAE map on the side.*

- Once trained, we throw away the encoder. To generate, we sample a random z from the standard normal distribution and pass it through the decoder. *(Re-roll.)*
- Because KL organized the space around that distribution, we get a plausible new cat. Soft, but a cat.
- The plain autoencoder, sampled the same way, gives tiger-stripe texture: its codes never lived near N of zero and one, so we're sampling the countryside. Even with its own mean and spread you get collages, because the space between its codes was never trained to be cats.
- These two models are bigger than the ones running live, so their samples are precomputed. *(Click the small map.)* The tiny live map is the small VAE: every point on it is a cat. No monsters.

*Slide #s27: interpolation. Pick A and B; three rows of 8 frames: a pixel crossfade, the bigger plain AE's walk, the bigger VAE's walk; a scrubber with a big frame per model.*

- We can also interpolate: encode two cats and walk the straight line between their two codes, decoding as we go.
- **Analogy:** it's the difference between a crossfade and a morph in a video editor. Top row: fading the pixels just stacks two ghost cats. That's why we walk the code, not the pixels.
- The plain AE keeps one cat, but look at the middle frames: smears and glitches, because the points between its codes were never trained.
- The VAE morphs smoothly from one cat to the other. Softer, but every frame is a cat, because KL made the whole region around the codes decodable.

## Part 6: Limitations, extensions, and wrap-up

### Limitations

*Slide #s28: blur. Loupe over original vs reconstruction, and the pixel average of 1 up to all 29,331 training cats.*

- VAEs aren't perfect. The biggest issue is blur. Partly because of MSE. If there are several possible sharp answers, MSE is minimized by averaging them, and an average of sharp images is a blurry image.
- **Analogy:** if you're not sure whether a friend will arrive from the left door or the right door, the answer with the smallest average squared error is to point at the wall in the middle. Safe on average, right never. *(Slide the averaging slider all the way.)* Average a few cats and you get a soft cat. Average all 29,331 training cats and you get one beige ghost: the safest possible guess, and the opposite of sharp.
- It's also partly because of the KL term, which caps how much detail can be squeezed into the latent space.

### Extensions

*Slide #s29: β-VAE, VQ-VAE, and Stable Diffusion. "Remember this?" opens a note about the KL dial. It does not leave the slide.*

- People have built a lot on top. Briefly:
- A β-VAE, or beta-VAE, adds a weight called beta to the KL term. A higher beta gives a more disentangled latent space, at the cost of reconstruction quality. (Disentangled means each latent dimension controls one separate feature. Remember the faces: one axis for smile, one for hair. That's the dream a β-VAE chases.)
- A conditional VAE fixes another limitation: a plain VAE can't take requests. Give the encoder and decoder a label too, and you can ask for a tabby.
- A VQ-VAE, Vector Quantized VAE, uses a learned dictionary of discrete codes instead of a continuous cloud. **Analogy:** a paint-by-numbers kit with a fixed palette instead of mixing any colour. It's sharper, and a close cousin, the discrete VAE or dVAE, powered the first DALL-E.
- And VAEs are still inside modern image generation. Stable Diffusion uses a VAE to compress a 512 by 512 image into a 64 by 64 by 4 latent, 48 times fewer numbers.
- Diffusion runs inside that latent space, which is much faster, so when you generate an image with Stable Diffusion, the very last step is a VAE decoder turning latents back into pixels.

### How VAEs compare to other image generation techniques

*Slide #s30: a comparison table of VAEs, GANs and diffusion, revealed row by row, the trilemma triangle, and real samples from a bigger VAE, a GAN (Gary) and a small diffusion model.*

|  | VAE | GAN | Diffusion |
| --- | --- | --- | --- |
| Image quality | Blurry | Sharp | Sharp |
| Variety of outputs | High | Can suffer mode collapse | High |
| Generation speed | Fast, one pass | Fast, one pass | Slow, many steps |
| Training | Stable | Unstable | Stable |
| Encoder for real images | Yes | No | No (noising process instead) |

- So how do VAEs compare to the two big alternatives: GANs, which we'll see next, and diffusion models?
- Without stealing the next presentation: a GAN has a generator that creates images and a discriminator that tries to tell real images from fake ones. **Analogy:** a forger and a detective, each getting better because of the other.
- Where is the VAE better? First, training is stable. We're minimizing one loss, and it goes down reliably. GANs are famously hard to train, because two networks are competing and the balance can easily break.
- Second, VAEs cover the whole dataset, because the reconstruction loss makes them represent every training image. GANs can suffer from mode collapse: the forger finds one painting that fools the detective and paints it forever.
- Third, a VAE has an encoder. We can take any real image and find its point in the latent space, which is what gives us interpolation, compression and anomaly detection. A standard GAN only goes from noise to image.
- So where is the VAE worse? Mainly image quality. GANs produce much sharper images, because the detective punishes anything that looks fake, including blur.
- And diffusion models produce both high quality and high variety, but they need many denoising steps, so they're much slower than a VAE or a GAN.
- *(Bottom row, "new samples".)* These are real samples from three models I trained on the same cats: our bigger VAE, a GAN, and a small diffusion model. The VAE is blurry; the GAN is sharp in one pass; diffusion is sharp and varied, but took 100 denoising steps per image.
- This GAN is Gary. Our Gary didn't collapse, thanks to modern tricks like spectral normalisation, but GANs can, and when they do you get the same few cats over and over.
- **Analogy:** it's the old "fast, good, cheap: pick two". Researchers call it the generative learning trilemma: quality, variety, speed. VAEs give variety and speed. GANs give quality and speed. Diffusion gives quality and variety.
- In practice the best systems combine them: the VAE inside Stable Diffusion is trained with a discriminator borrowed from GANs, to make its outputs sharper. They're building blocks, not rivals.

### Measuring it, and using it well

*Slide #s30b: three cards. "How do we measure it?" with live numbers computed in the browser on the 64 held-out cats; "Where it's used"; "Use it responsibly".*

- Two questions for any model: how do you measure it, and should you use it?
- *(Point at the live numbers.)* For reconstruction: PSNR, peak signal-to-noise ratio, on the 64 cats the models never trained on, computed right now. Higher is better, and it climbs with the size of the code. (Every 3 decibels is roughly half the squared error.) If the VAE scores below a plain autoencoder with fewer numbers, as it does tonight, that's the price of the KL leash: the blur slide in one number. For generation you can't compare to an answer, because there is no answer. The standard metric is FID, the Fréchet Inception Distance: run real and generated images through a pretrained network and compare the two clouds of features. Lower is better. To be honest, this deck shows reconstruction error, not FID.
- Where it's used: anomaly detection, denoising, compression, synthetic data for small datasets, and Stable Diffusion's latent space.
- Using it responsibly. A generative model learns its training data, including its biases. Ours only ever saw front-facing, centred cat faces, so a side-on cat looks like an anomaly to it. Swap cats for people and that becomes a fairness problem: an anomaly detector that flags people who weren't in the training set. And the same tools that imagine cats can imagine faces that never existed, so deepfakes, consent for face data and labelling synthetic images all matter.

### Wrap-up

*Slide #s31: a timeline in date order. Autoencoder, CNN, VAE, GAN, diffusion. Each row is a main point, revealed on the next press.*

- So to wrap up, as a timeline. *(Press per row.)* We started with CNNs, the judges: they tell images apart, they don't draw them.
- A plain encoder-decoder would just learn to copy the input, so we added a bottleneck: the carry-on.
- That gave us the autoencoder, which learns its own chart, like the faces chart, without any labels, by minimizing the reconstruction loss.
- But its map only has cities and monsters, so sampling from it doesn't give us new images.
- The VAE fixes this by giving every cat a neighbourhood instead of an address, and putting every neighbourhood on a KL leash, which makes the latent space smooth and continuous.
- The reparameterization trick, rolling the dice first, is what makes it trainable.
- And the result is a model we can sample new cats from, which still sits inside today's best generators.
- Thank you, and I'm happy to take any questions. And the next presenter will show you the forger and the detective properly.

*Slide #s32: Questions. Leave it up during Q&A.*

## Appendix A: analogy cheat sheet

| Slide | Idea | Analogy |
| --- | --- | --- |
| #s03 | Discriminative vs generative | A cat-show judge vs an artist |
| #s05 | Identity trap | Bringing the textbook into the exam: you copy, you don't learn |
| #s06b | Latent space | Describe a face with two numbers (smile, hair); everyone becomes a dot on a chart (Deepia) |
| #s07 | Pixels as input | 3Blue1Brown's 784-number digit, as a 12,288-number cat |
| #s08 | Bottleneck | Packing for two weeks with one carry-on |
| #s09 | High-dimensional spaces | A shadow on the wall: you can't tell a mug from a donut |
| #s10 | Self-supervised | The question sheet is also the answer key |
| #s11 | Training | A ball rolling downhill; which nudge helps most (3Blue1Brown) |
| #s12 | PCA vs AE | A straight ruler vs a tape measure |
| #s13 | Anomaly detection | A security guard who only knows the regulars |
| #s14 | Sparse AE | An orchestra where only a few players may play each song |
| #s15 | Holes in latent space | An old map: cities, and "here be monsters" |
| #s17 | Marginals, conditionals | A spreadsheet of grades and ages: column totals and one rescaled row |
| #s18 | Prior and decoder | A translator between cat photos and a simple invented language |
| #s19 | Encode a distribution | An exact address vs a neighbourhood |
| #s20 | KL term | A leash tying each dog to the centre of the park |
| #s21 | β dial | A tug of war between rebuilding and packing together |
| #s23 | Chain rule | A relay of blame, passed back one layer at a time |
| #s24 | Sampling breaks gradients | You can't ask a dice how to roll higher |
| #s25 | Reparameterization | Roll the dice first, then stretch by σ and move by μ |
| #s27 | Interpolation | Crossfade vs morph |
| #s28 | Blur | Pointing at the wall between two doors |
| #s29 | VQ-VAE | Paint-by-numbers with a fixed palette |
| #s30 | GAN / trilemma | A forger and a detective; fast, good, cheap: pick two |

## Appendix B: how this maps to the subject outline

- LO1 (structure of visual data, suitable approaches): an image as 12,288 numbers (#s07), discriminative vs generative (#s03), and when a reconstruction model suits a task (#s13).
- LO2 (design and implement CNNs and generative models): the conv encoder and transposed-conv decoder (#s04), the model families trained for this deck (live in the browser), plus a bigger VAE, a GAN and a small diffusion model for the comparisons.
- LO3 (pre-trained and diffusion models): the VAE inside Stable Diffusion (#s29), and how VAEs, GANs and diffusion fit together (#s30).
- LO4 (evaluate and compare with appropriate metrics): reconstruction error and PSNR on held-out cats, PCA vs AE (#s12), FID and its caveats (#s30b).
- LO5 (generative AI in production): anomaly detection, denoising, compression and synthetic data (#s13, #s30b); the deck itself is a deployed ONNX Runtime Web model (week 10).
- LO6 (communicate rigorously): cited history (#s06, #s16), honest limitations (blur, blind spots, which samples are live and which are precomputed), and responsible use (#s30b).
