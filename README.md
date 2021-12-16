# PokeGAN
PokeGAN is a GAN made for creating Pokemon from coloring pictures. </br>
A GAN is a Generative Adversarial Network, which works with a Generator and a Discriminator. The generator generates images based on the feedback of the discriminator. The discriminator gets the input of the generator as well as the original images and has to decide if the input is fake or not. If a generated image is decided as fake, the generator tries to improve to get a non fake labeled image generated. With GANs it is possible to create Deep Fakes.</br></br>
Here 34 pictures were used to train two kind of GANs: Dense and CNN.</br>
The class offers the opportunity to save the model as well as screenshots every desired epoch.</br>
Two kind of images are generated depending on the noise as input for the generator:</br>
Random images using random created noise for every image. Via random_seed the images can become comparable.</br>
Range images using the same range of noise for every image. Thus they are comparable.</br>
Additionally a gif can be created for both kinds of generated images.
