# Autoencoders & VAEs: Presentation Script

Tuesday 20 October 2026 · @Danish

Expect about 24 minutes of speaking, or 29 minutes with slide changes and pauses, before questions. Lines in italics are slide cues, not things to say. Notes in brackets are short explanations you can say out loud or skip.

| Part | Approx. words | Approx. time |
| --- | --- | --- |
| 1. From classification to generation | 430 | 3.5 min |
| 2. Autoencoders | 580 | 4.5 min |
| 3. Other types of autoencoders | 260 | 2 min |
| 4. Why autoencoders can't generate | 240 | 2 min |
| 5. Variational autoencoders | 760 | 6 min |
| 6. Limitations, extensions, comparison, wrap-up | 820 | 6.5 min |
| Total | \~3,100 | \~24 min |

Times assume a steady presenting pace of about 130 words per minute. If you need to cut, Part 3 and the Extensions in Part 6 are the safest to shorten. Keep the comparison, since it sets up the GAN presentation.

## Part 1: From classification to generation

*Slide: title. Tuesday 20 October 2026. No cats on this one.*

*Slide: warm-up. "Cat people or dog people?" Ask the room to raise hands. Count them yourself; the slide does not keep a tally.*

*Slide: a CNN classifier taking an image and outputting "cat".*

- So far in this course, almost everything we've looked at has been about understanding images.
- We've looked at CNNs, ResNet, MobileNet, Vision Transformers, CvT and a bunch of other transformers. (CvT stands for Convolutional vision Transformer, which mixes convolutions into the transformer.)
- All of these take in an image and output something about it, like a class label.
- These are called discriminative models. (A discriminative model learns the probability of a label given an image. In other words, it learns to tell things apart.)
- But what if we want to go the other way and actually create images?
- That's what generative models do. (A generative model learns something about the distribution of the images themselves, so it can produce new ones.)
- Today we're going to look at one of the first and most important families of generative models, which are autoencoders and variational autoencoders.

*Slide: an encoder-decoder diagram. Flipping the switch plays the layers from the input out to the ending. There is no "next layer" button.*

- So the first question is, could we just use the networks we already have?
- A normal classification CNN won't work as it is, because it outputs class scores, not an image.
- But we could take the convolution and pooling layers we already know and use them to process the image.
- Then we add a second half that does the reverse, and grows the image back up to its original size.
- The first half is called the encoder, and the second half is called the decoder.
- To grow the image back up, the decoder uses transposed convolutions or upsampling layers. (A transposed convolution is basically a convolution in reverse. Instead of shrinking the image, it makes it bigger.)
- So now we have a network that takes an image in and gives an image out.

*Slide: the same diagram, but with every layer as wide as the input.*

- But there's a problem.
- If every layer in this network is as big as the input or bigger, the network can just learn to copy the input straight to the output.
- This is called learning the identity function. (The identity function is just a function where the output equals the input.)
- The loss would be perfect, but the model wouldn't have learned anything useful.
- It wouldn't know anything about edges, shapes, or what makes a face look like a face.
- And because it hasn't learned any features, it can't help us understand or generate new images.
- So how do we tackle this?
- We need to force the model to learn features.

## Part 2: Autoencoders

*Slide: a short history. 1986 Rumelhart, Hinton & Williams; 2006 Hinton & Salakhutdinov; 2008 Vincent et al. on denoising.*

- Before the demo, a bit of history, and only the dates I'm confident about.
- The usual citation for autoencoders is Rumelhart, Hinton and Williams in 1986: train a network to rebuild its input so the hidden layer learns a code.
- The deep version that people actually used on images is Hinton and Salakhutdinov, Science, 2006.
- Denoising autoencoders are Vincent, Larochelle, Bengio and Manzagol, 2008.

*Slide: the encoder-decoder diagram with a narrow middle section labelled "bottleneck" and "z".*

- Autoencoders force the model to learn features by reducing the dimensions between the encoder and the decoder.
- This narrow middle part is called the bottleneck.
- The bottleneck is smaller than the input and the output.
- So the model can't just copy every pixel through. There isn't enough room.
- Instead, it has to squeeze the image into a small set of numbers that capture the most useful information.
- This small set of numbers is called the latent vector, and we usually call it z.
- The space that all these latent vectors live in is called the latent space. (Latent just means hidden. These are features the model discovers itself, we never label them.)
- Because the bottleneck is smaller than the input, this is called an undercomplete autoencoder.
- So to recap the structure: the encoder compresses the image down to z, and the decoder tries to rebuild the original image from z alone.

*Slide: input image, output image, and the loss between them.*

- Now how do we train this?
- We compare the output to the input and get a loss function.
- This is called the reconstruction loss, because it measures how well the model reconstructed the input.
- The most common choice is MSE. (MSE stands for Mean Squared Error. It's the average of the squared differences between each output pixel and the matching input pixel.)
- Another option is BCE, which works well when pixel values are between 0 and 1. (BCE stands for Binary Cross-Entropy. It treats each pixel as a probability.)
- Then we train the model with backpropagation to reduce this loss, like any other network.
- One important thing to notice here is that we don't need any labels.
- The input is the target. The image is its own answer.
- That makes this self-supervised learning. (Self-supervised means the training signal comes from the data itself, not from human labels.)
- That's a big deal, because unlabelled images are everywhere and labelled ones are expensive.

*Slide: PCA versus autoencoder.*

- I want to be a bit more precise about what "most important features" means.
- The model learns whatever features are most useful for reconstructing the image under our loss function.
- There's a nice connection here to something you may already know, which is PCA. (PCA stands for Principal Component Analysis. It's a classic technique that reduces data to the few directions where it varies the most.)
- If you build an autoencoder with only linear layers and train it with MSE, it learns the same subspace as PCA.
- So you can think of a deep autoencoder as a nonlinear version of PCA.
- The nonlinear activations are what let it capture much more complex structure, like curves and textures, that PCA can't.

*Slide: four tabs, each one live. Compression, denoising, features, anomaly detection.*

- So what are autoencoders actually used for?
- Compression, because the latent vector is a much smaller representation of the image. The slider is the same family of models as the bottleneck, ae_2 through ae_256.
- Denoising, which is cleaning up noisy or damaged images. Here the noise slider feeds dae_128.
- Feature extraction, where we throw away the decoder and use the encoder's output as input to another model. The map shows ae_2 codes, and the four neighbours are real decodes of nearby codes.
- And anomaly detection.
- For anomaly detection, we train the autoencoder only on normal data.
- When something unusual comes in, the model reconstructs it badly, because it has never seen anything like it.
- So a high reconstruction error flags an anomaly. This is used for things like spotting defects on a factory line.

## Part 3: Other types of autoencoders

*Slide: four tabs. Same job, one extra rule each: sparse, denoising, contractive, convolutional.*

- What we've built so far is the undercomplete autoencoder.
- Say that out loud: these are not four new jobs. They are four extra rules about the code.
- But the bottleneck isn't the only way to force the model to learn features.
- There are a few other variants worth knowing.
- A sparse autoencoder can have a large hidden layer, even larger than the input.
- But we add a penalty that forces most of the neurons to stay switched off for any one image.
- The penalty is usually L1 regularization, which punishes large activations, or a KL divergence term, which we'll explain properly later. (L1 regularization adds the sum of the absolute values of the activations to the loss.)
- Because only a few neurons can fire at once, each one has to learn a specific, meaningful feature.
- A denoising autoencoder adds noise to the input image but still asks the model to reconstruct the clean original.
- The model can't just copy the input anymore, because the input is corrupted.
- So it has to learn what a real, clean image looks like.
- A contractive autoencoder penalizes the model if small changes in the input cause big changes in the latent vector.
- This makes the learned features more stable and robust to small variations.
- And finally, a convolutional autoencoder is just an autoencoder that uses convolution layers in the encoder and transposed convolutions in the decoder.
- This is what we'd actually use for images, and it's what I showed in the diagrams earlier.
- You can also combine these. For example, a convolutional denoising autoencoder is very common in practice.

## Part 4: Why autoencoders can't generate images

*Slide: a 2D latent space plot of MNIST digits, with separate clusters and big empty gaps. (MNIST is the classic dataset of handwritten digits 0 to 9.)*

- So now we have a model that learns features. But can we generate new images with it?
- The idea would be to throw away the encoder, pick a random point in the latent space, and pass it through the decoder.
- If the decoder has learned what images look like, it should give us a brand new image.
- But in practice, this mostly gives us garbage.
- To see why, look at what the latent space actually looks like after training.
- The encoder maps each training image to a single point.
- Those points end up in scattered clusters, with big empty gaps between them.
- The decoder has never seen anything from those gaps, so it has no idea what to output there.
- On top of that, the latent space is irregular.
- We don't know where the valid regions are, how spread out they are, or what range of values we should sample from.
- So if we pick a random point, we'll very likely land in a gap and get nonsense.
- So the bottleneck solved the feature learning problem, but it didn't solve the generation problem.
- So how do we tackle this?
- We need the latent space to be continuous and organized, so that any point we pick decodes into something meaningful.
- And that's exactly what variational autoencoders do.

## Part 5: Variational autoencoders (VAEs)

*Slide: where VAEs come from. Kingma & Welling 2013, and Rezende, Mohamed & Wierstra 2014.*

- Two groups, almost at once.
- Kingma and Welling, December 2013, then ICLR 2014: Auto-Encoding Variational Bayes.
- Rezende, Mohamed and Wierstra, ICML 2014: the same idea, worked out independently.
- Don't merge them into one person or one date.

### The core idea

*Slide: VAE diagram. Encoder outputs two vectors, mu and sigma, then a sampling step, then z, then the decoder.*

- Instead of encoding each image to a single point, the encoder in a VAE outputs a probability distribution.
- Specifically, it outputs two vectors. A mean, called mu, and a standard deviation, called sigma.
- Together these describe a Gaussian distribution. (A Gaussian is the normal distribution, the bell curve.)
- We then sample a point z from that distribution and pass it to the decoder.
- So the same image maps to a small cloud in the latent space instead of a single point.
- Every time we pass the same image through, we get a slightly different z, but the decoder still has to rebuild the same image.
- So the decoder learns that the whole area around each point should produce a similar image.
- This already starts to fill in the gaps.

### The loss function

*Slide: Loss = reconstruction loss + KL divergence.*

- The VAE loss has two parts.
- The first part is the reconstruction loss, the same as before. It makes sure the output still looks like the input.
- Again, this is usually MSE (Mean Squared Error) or BCE (Binary Cross-Entropy).
- The second part is the KL divergence. (KL stands for Kullback–Leibler. KL divergence measures how different one probability distribution is from another. If they're identical, it's zero.)
- The KL term pushes each encoded distribution to be close to a standard normal distribution, which has a mean of 0 and a standard deviation of 1.
- For one dimension the closed form is one half of sigma squared plus mu squared, minus 1, minus the log of sigma squared. The training loss sums that over every latent dimension.
- This pulls all the distributions toward the center of the latent space and makes them overlap, which removes the gaps.
- It also stops the model from cheating by making sigma tiny, which would turn it back into a normal autoencoder.

*Slide: two 2D latent space plots side by side. Left: reconstruction loss only, scattered clusters with gaps. Right: with KL, clusters packed together around zero.*

- The two terms actually work against each other.
- The reconstruction loss wants each image to have its own distinct spot, so it can be rebuilt accurately.
- The KL loss wants everything bunched together around zero.
- The balance between them gives us a latent space that is both meaningful and smooth.
- Similar images end up close together, and there are no empty gaps in between.
- Formally, this combined loss is the negative of something called the ELBO. (ELBO stands for Evidence Lower Bound. It's a lower bound on the likelihood of the data, meaning how probable the model thinks real images are.)
- We'd ideally like to maximize that likelihood directly, but calculating it is intractable. (Intractable means it's impossible to compute in practice.)
- So instead we maximize the ELBO, which is a practical stand-in that pushes the likelihood up.
- One caveat: our reconstruction term is mean squared error. For a decoder with a fixed variance, that matches the negative log-likelihood up to a constant and a fixed scale.

### Backprop, then the reparameterization trick

*Slide: forward, loss, chain rule. A plain autoencoder is differentiable the whole way.*

- Backpropagation is just the chain rule, walked backwards through the layers.
- Each conv layer only needs the gradient from the layer above it.
- A plain autoencoder has no problem here. z is a function of the weights, so the gradient flows through it.

*Slide: plain AE on the left, z = encoder(x). VAE on the right, z sampled from N(mu, sigma).*

- The break is only the sampling step in a VAE.
- mu and sigma are ordinary numbers. You can differentiate those.
- A random draw is not a smooth function of them, so the gradient stops at the dice.

*Slide: z = mu + sigma × epsilon, with epsilon drawn from a standard normal.*

- There's one problem with training this.
- We train neural networks with backpropagation, which needs gradients to flow through every step.
- But sampling a random value isn't a differentiable operation, so gradients can't flow back through it to the encoder.
- So how do we fix this?
- We move the randomness outside the network.
- We sample a random value called epsilon from a standard normal distribution.
- Then we calculate z as mu plus sigma times epsilon.
- Now mu and sigma are just regular values in a calculation, so gradients can flow through them. Epsilon is treated as a fixed input.
- This is called the reparameterization trick, and it's what makes VAEs trainable.
- A small practical note: the encoder usually outputs the log of the variance instead of sigma directly. (Variance is sigma squared.) This keeps the numbers stable and makes sure the variance is always positive.

### Generating new images

*Slide: a grid of generated digits or faces, plus an interpolation strip between two images.*

- Once the VAE is trained, we can throw away the encoder.
- To generate a new image, we sample a random z from the standard normal distribution and pass it through the decoder.
- Because the KL term organized the latent space around that distribution, the output is a plausible new image.
- We can also interpolate.
- That means taking two images, encoding them, and walking along the straight line between their two latent points.
- The decoder produces the frames along that line.
- A straight walk between two real codes stays near the data even for a plain autoencoder, so those two rows look similar. The plain autoencoder's failure was the random samples we just saw, not this walk.

## Part 6: Limitations, extensions, and wrap-up

### Limitations

*Slide: a VAE output next to the original, showing the blur.*

- VAEs aren't perfect. The biggest issue is that their outputs tend to be blurry.
- This is partly because of MSE. If there are several possible sharp answers, MSE is minimized by averaging them, and an average of sharp images is a blurry image.
- It's also partly because of the trade-off with the KL term, which limits how much detail can be squeezed into the latent space.

### Extensions

*Slide: β-VAE, VQ-VAE, and Stable Diffusion. "Remember this?" opens a note about the KL dial. It does not leave the slide.*

- People have built a lot on top of the basic VAE. I'll mention two.
- A β-VAE, or beta-VAE, adds a weight called beta to the KL term.
- A higher beta gives a more organized and disentangled latent space, at the cost of reconstruction quality. (Disentangled means each latent dimension controls one separate feature, like rotation, size, or brightness.)
- A VQ-VAE, which stands for Vector Quantized VAE, uses a learned dictionary of discrete codes instead of a continuous distribution.
- This produces sharper results, and it was used in the first version of DALL-E.
- And VAEs are still a core part of modern image generation today.
- Stable Diffusion, which we'll cover with generative models, uses a VAE to compress images into a smaller latent space.
- The diffusion process then runs inside that latent space instead of on full-size images, which makes it much faster.
- So when you generate an image with Stable Diffusion, the very last step is a VAE decoder turning latents back into pixels.

### How VAEs compare to other image generation techniques

*Slide: a comparison table of VAEs, GANs, and diffusion models.*

|  | VAE | GAN | Diffusion |
| --- | --- | --- | --- |
| Image quality | Blurry | Sharp | Sharp |
| Variety of outputs | High | Can suffer mode collapse | High |
| Generation speed | Fast, one pass | Fast, one pass | Slow, many steps |
| Training | Stable | Unstable | Stable |
| Encoder for real images | Yes | No | No (noising process instead) |

- So how do VAEs compare to the other ways of generating images?
- The two big alternatives are GANs, which we'll see in the next presentation, and diffusion models.
- GANs stand for Generative Adversarial Networks. (Adversarial means two networks competing against each other.)
- Without stealing the next presentation, the basic idea is that a GAN has a generator that creates images and a discriminator that tries to tell real images from fake ones.
- Instead of a reconstruction loss, the generator learns by trying to fool the discriminator.
- So where is the VAE better?
- First, VAE training is stable. We're just minimizing one loss function, and it goes down reliably.
- GANs are famously hard to train, because two networks are competing and the balance between them can easily break.
- Second, VAEs cover the whole dataset.
- GANs can suffer from something called mode collapse. (Mode collapse is when the generator finds a few outputs that fool the discriminator and keeps producing only those, so you lose variety.)
- The KL term and the reconstruction loss in a VAE force it to represent every training image, so it doesn't have this problem.
- Third, a VAE has an encoder.
- That means we can take any real image and find its point in the latent space, which is what lets us do interpolation, compression, and anomaly detection.
- A standard GAN only goes from noise to image. It has no built-in way to go back.
- So where is the VAE worse?
- Mainly in image quality. GANs produce much sharper images, because the discriminator punishes anything that looks fake, including blur.
- And diffusion models go even further, producing both high quality and high variety.
- But diffusion has its own cost. It generates an image over many denoising steps, so it's much slower than a VAE or a GAN, which each need just one pass.
- The triangle keeps all three labels up at once: who is strong on quality, who on variety, who on speed.
- Gary is not a GAN photo. The repo has no GAN model. Those nine pictures are our VAE decoding one code, to show mode collapse. Say that. A real GAN would be sharper.
- Researchers sometimes call this the generative learning trilemma. (A trilemma is a choice between three options where you can only fully get two.)
- The three things we want are high quality, high variety, and fast generation.
- VAEs give us variety and speed. GANs give us quality and speed. Diffusion gives us quality and variety.
- And in practice, the best systems combine them.
- The VAE inside Stable Diffusion is actually trained with a discriminator, borrowed from GANs, to make its outputs sharper.
- So these techniques aren't really rivals. They're building blocks that are often used together.
- And that's a good lead into the next presentation, where we'll see exactly how GANs work.

### Wrap-up

*Slide: a timeline. CNN, autoencoder, VAE, GAN, diffusion. Each row is a main point, revealed on the next press.*

- So to wrap up, as a timeline, not a footnote.
- CNNs, from LeCun in 1989 and 1998 through AlexNet in 2012, tell images apart. They do not draw them.
- We started with discriminative models that understand images, and asked how we could generate them instead.
- A plain encoder-decoder would just learn to copy the input, so we added a bottleneck.
- That gave us the autoencoder, which learns useful features without any labels, by minimizing the reconstruction loss.
- But its latent space has gaps, so sampling from it doesn't give us new images.
- The VAE fixes this by encoding to a distribution and adding a KL divergence term, which makes the latent space smooth and continuous.
- The reparameterization trick is what makes it trainable.
- And the result is a model we can sample from to generate new images, and which still sits inside today's state-of-the-art generators.
- Thank you, and I'm happy to take any questions.
