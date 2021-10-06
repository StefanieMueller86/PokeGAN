"""
Created by Stefanie Müller, October 2021
Based on my deep learning course project from september 2021.
"""
import os
import shutil

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

from tqdm import tqdm

import numpy as np
import math

import matplotlib.pyplot as plt

from PIL import Image
import glob


class PokeGAN:
    """
    PokeGAN is a GAN made for creating Pokemon from coloring pictures.\n
    Here 34 pictures were used to train two kind of GANs: Dense and CNN.\n
    The class offers the opportunity to save the model as well as screenshots every desired epoch.\n
    Two kind of images are generated depending on the noise as input for the generator:\n
    Random images using random created noise for every image. Via random_seed the images can become comparable.\n
    Range images using the same range of noise for every image. Thus they are comparable.\n
    Additionally a gif can be created for both kinds of generated images.
    """

    def __init__(self,
                 gantype,
                 n_all_images=34,
                 save_model=False,
                 save_path=None,
                 save_directory=None,
                 save_images=False,
                 show_image_every_epoch=100,
                 save_image_every_epoch=500,
                 save_model_every_epoch=500):
        """Initializes the PokeGAN.
        Two GANs are predefined: "CNN" and "Dense".
        There are several options to ensure images and models are saved after desired epoch.
        The folder structure will be generated during the init. Before overwriting/deleting existing folders the user will be asked.

        :param gantype: Predefined are "CNN" and "Dense" GAN.
        :param n_all_images: Total number of Images.
        :param save_model: if true, the Model will be saved after save_model_every_epoch epoch.
        :param save_path: if None, workingdirectory will be taken.
        :param save_directory: if None, folder "PokeGAN_Outputs_{gantype}" will be created
        :param save_images: if true, Images will be saved after save_image_every_epoch epoch.
        :param show_image_every_epoch: Plots an image every Epoch chosen.
        :param save_image_every_epoch: Saves an image every Epoch chosen.
        :param save_model_every_epoch: Saves the model every Epoch chosen. All Models in one run will be kept.
        """
        self.gantype = gantype
        if not (self.gantype == 'CNN' or self.gantype == 'Dense'):
            raise ValueError('Valid GAN types are: "CNN" or "Dense"')

        self.n_all_images = n_all_images
        self.all_images_normalized = []
        self.images = []

        self.optimizer = None
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.latent_dim = 0
        self.target_image_size = 0

        # Creating save directories
        if save_model or save_images:
            if save_directory is None:
                save_directory = f"PokeGAN_Outputs_{gantype}"
            if save_path is None:
                save_path = os.getcwd()

            self.full_save_path = f"{save_path}/{save_directory}/"
            self.full_save_path_random_images = f"{self.full_save_path}/RandomImages/"
            self.full_save_path_range_images = f"{self.full_save_path}/RangeImages/"

            if not os.path.exists(self.full_save_path):
                os.makedirs(self.full_save_path)
            else:
                user_input = input(
                    f"!ATTENTION!\nDesired folder already exists: {self.full_save_path}\nDo you want to delete _ALL_ saved data in that folder?\nUser input (y=yes/n=no): ")
                if user_input == 'y':
                    print("Program proceeds. All Saved Data in desired Folder will be deleted.")
                    shutil.rmtree(self.full_save_path)
                    os.makedirs(self.full_save_path)
                elif user_input == 'n':
                    raise Exception("Program aborted. Please define a new PokeGan with an empty directory.")

            if not os.path.exists(self.full_save_path_random_images):
                os.makedirs(self.full_save_path_random_images)
            if not os.path.exists(self.full_save_path_range_images):
                os.makedirs(self.full_save_path_range_images)

        self.save_model = save_model
        self.save_images = save_images
        self.show_image_every_epoch = show_image_every_epoch
        self.save_image_every_epoch = save_image_every_epoch
        self.save_model_every_epoch = save_model_every_epoch

    def get_save_paths(self):
        """
        Returns the save paths.
        :return: 3 variables will be returned, containing the save paths for all outputs, randomImages and rangeImages.
        """
        print("Save Directory: ", self.full_save_path)
        return self.full_save_path, self.full_save_path_random_images, self.full_save_path_range_images

    def load_data(self, datapath, batch_size=None, target_image_size=128):
        """
        Loads images from directories via flow_from_directory.
            Includes Resizing and rescaling.
            No Data Augmentation Options are activated.
            Normalizes the images afterwards.
        :param datapath: Parentfolder where the images are located. Subfolders are allowed.
        :param batch_size: Batch_size for flow_from_directory.
        :param target_image_size: Target_image_size for resizing Original Images.
        :return: Resized Images will be returned.
        """
        if batch_size is None:
            batch_size = self.n_all_images

        self.target_image_size = target_image_size

        datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                               validation_split=0)

        image_data = datagen.flow_from_directory(
            datapath,
            target_size=(target_image_size, target_image_size),
            color_mode='grayscale',
            batch_size=batch_size,
            shuffle=True,
            seed=None,
            save_to_dir=None,
            follow_links=False,
            subset='training',
            interpolation='nearest'
        )

        self.images = image_data.next()

        if self.gantype == 'CNN':
            all_images = self.images[0][:]
        elif self.gantype == 'Dense':
            all_images = self.images[0][:].reshape(-1, target_image_size * target_image_size)

        # Normalizing between -1 to +1
        self.all_images_normalized = all_images.astype('float32') * 2 - 1

        return self.images

    def plot_images(self, n_pictures=None):
        """
        Creates a plot of up to all original images.
        Images will be saved if specified on instantiating the PokeGAN.
        :param n_pictures: Number of pictures to be shown in the plot. It is recommended to reduce this number if there are more than 50 original images.
        """

        if n_pictures is None:
            n_pictures = self.n_all_images

        columns = math.ceil(math.sqrt(n_pictures))
        rows = math.ceil(math.sqrt(n_pictures))

        for i in range(1, n_pictures + 1):
            plt.subplot(columns, rows, i)
            img = self.images[0][i - 1]
            plt.imshow(img, cmap='gray')
            plt.title(i)
            plt.axis('off')

        if self.save_images:
            plt.savefig(f"{self.full_save_path}Original_Images")

        plt.show()
        plt.close()

    def cnn_generator_create(self, leaky_factor, latent_dim):
        """
        Creates a CNN Generator for the GAN.

        :param leaky_factor: Means the alpha value.
        :param latent_dim: Latent Dimension to initialise value space of the generator.
        :return: CNN Generator will be returned.
        """
        self.latent_dim = latent_dim

        generator = Sequential(name='Generator')

        generator.add(keras.layers.Dense(8 * 8 * 3, input_shape=(latent_dim,)))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.LeakyReLU(leaky_factor))

        generator.add(keras.layers.Reshape((8, 8, 3)))

        # 8*8 -> 16*16
        generator.add(keras.layers.Conv2DTranspose(128, (8, 8), strides=(2, 2), padding='same', use_bias=False))
        assert generator.output_shape == (None, 16, 16, 128)
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.ReLU())

        # 16*16 -> 32*32
        generator.add(keras.layers.Conv2DTranspose(32, (8, 8), strides=(2, 2), padding='same', use_bias=False))
        assert generator.output_shape == (None, 32, 32, 32)
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.ReLU())

        # 32*32 -> 64*64
        generator.add(keras.layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        assert generator.output_shape == (None, 64, 64, 16)
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.ReLU())

        # 64*64 -> 128*128
        generator.add(
            keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert generator.output_shape == (None, 128, 128, 1)

        self.generator = generator
        return generator

    def dense_generator_create(self, leaky_factor, latent_dim):
        """
        Creates a Dense Generator for the GAN.

        :param leaky_factor: Means the alpha value.
        :param latent_dim: Latent Dimension to initialise value space of the generator.
        :return: Dense Generator will be returned.
        """
        self.latent_dim = latent_dim

        generator = Sequential(name='Generator')
        generator.add(keras.layers.Dense(256, input_shape=(latent_dim,)))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.LeakyReLU(leaky_factor))

        generator.add(keras.layers.Dense(512))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.LeakyReLU(leaky_factor))

        generator.add(keras.layers.Dense(1024))
        generator.add(keras.layers.BatchNormalization())
        generator.add(keras.layers.LeakyReLU(leaky_factor))

        generator.add(keras.layers.Dense(self.target_image_size * self.target_image_size, activation='tanh'))

        generator.summary()

        self.generator = generator
        return generator

    def generator_create(self, leaky_factor=0.2, latent_dim=2):
        """
        Function to easily chose the generator depending on the chosen gantype when instantiating the PokeGAN.
        :param leaky_factor: Means the alpha value.
        :param latent_dim: Latent Dimension to initialise value space of the generator. Range-Image Output only possible with latent_dim=2.
        :return: Returns the generator based on the used function.
        """
        if self.gantype == 'CNN':
            self.cnn_generator_create(leaky_factor=leaky_factor, latent_dim=latent_dim)
        elif self.gantype == 'Dense':
            self.dense_generator_create(leaky_factor=leaky_factor, latent_dim=latent_dim)

    def cnn_discriminator_create(self, optimizer, dropout_rate, leaky_factor=0.2):
        """
        Creates a CNN Discriminator for the GAN.
        :param optimizer: Adam with learning rate 1e-4 works.
        :param dropout_rate: Dropout rate for all layers but the first. 0.5 works
        :param leaky_factor: Means the alpha value.
        :return: CNN Discriminator will be returned.
        """
        self.optimizer = optimizer

        discriminator = Sequential(name='Discriminator')

        # 128*128 -> 64*64
        discriminator.add(keras.layers.Conv2D(16, (8, 8), strides=(2, 2), padding='same',
                                              input_shape=[128, 128, 1]))
        discriminator.add(keras.layers.LeakyReLU())

        # 64*64 -> 32*32
        discriminator.add(keras.layers.Conv2D(32, (8, 8), strides=(2, 2), padding='same'))
        discriminator.add(keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        # 32*32 -> 16*16
        discriminator.add(keras.layers.Conv2D(64, (8, 8), strides=(2, 2), padding='same'))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        # 16*16 -> 8*8
        discriminator.add(keras.layers.Conv2D(128, (8, 8), strides=(2, 2), padding='same'))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        # 8*8 -> 4*4
        discriminator.add(keras.layers.Conv2D(256, (8, 8), strides=(2, 2), padding='same'))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        # 4*4 -> 2*2
        discriminator.add(keras.layers.Conv2D(512, (8, 8), strides=(2, 2), padding='same'))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        discriminator.add(keras.layers.Flatten())
        discriminator.add(keras.layers.Dense(1, activation='sigmoid'))

        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        discriminator.summary()

        discriminator.trainable = False

        self.discriminator = discriminator
        return discriminator

    def dense_discriminator_create(self, optimizer, leaky_factor, dropout_rate):
        """
        Creates a Dense Discriminator for the GAN.
        :param optimizer: RMSprop with learning rate 0.0008, clipvalue of 1.0 and decay of 1e-8 works.
        :param dropout_rate: Dropout rate for all layers. 0.2 works.
        :param leaky_factor: Means the alpha value.
        :return: Dense Discriminator will be returned.
        """
        self.optimizer = optimizer

        discriminator = Sequential(name='Discriminator')

        discriminator.add(keras.layers.Dense(1024, input_shape=(self.target_image_size * self.target_image_size,)))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        discriminator.add(keras.layers.Dense(512))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        discriminator.add(keras.layers.Dense(256))
        discriminator.add(keras.layers.LeakyReLU(leaky_factor))
        discriminator.add(keras.layers.Dropout(dropout_rate))

        discriminator.add(keras.layers.Dense(1, activation='sigmoid'))

        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        discriminator.summary()

        self.discriminator = discriminator
        return discriminator

    def discriminator_create(self, leaky_factor=0.2, dropout_rate=0.5):
        """
        Function to easily chose the discriminator depending on the chosen gantype when instantiating the PokeGAN.
        :param leaky_factor: Means the alpha value. 0.2 works.
        :param dropout_rate: Dropout rate for all layers but the first for CNN- and all for Dense-Discriminator. 0.2 works well for Dense and 0.5 for CNN.
        :return: Returns the discriminator based on gantype.
        """
        if self.gantype == 'CNN':
            self.cnn_discriminator_create(leaky_factor=0.2,
                                          dropout_rate=dropout_rate,
                                          optimizer=Adam(1e-4))
        elif self.gantype == 'Dense':
            self.dense_discriminator_create(leaky_factor=leaky_factor,
                                            dropout_rate=dropout_rate,
                                            optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8))

    def combine_gan(self, generator=None, discriminator=None):
        """
        Combines the GAN based on Generator and Discriminator.
        :param generator: if None it will be taken from the generator_create() function.
        :param discriminator: if None it will be taken from the discriminator_create() function.
        :return: Returns the GAN.
        """
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        discriminator.trainable = False

        gan_core = discriminator(generator(generator.input))
        self.gan = Model(inputs=generator.input, outputs=gan_core, name='GAN')
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.gan.summary()
        return self.gan

    def __create_images(self, epoch, target_image_size=None):
        """
        Creates images for random and range Images.
        :param epoch: Current epoch in the fit process.
        :param target_image_size: Target image size of the output.
        """
        if target_image_size is None:
            target_image_size = self.target_image_size

        # RandomImages
        dim = (10, 10)
        # np.random.seed(42)
        noise = np.random.normal(0, 1, size=[100, self.latent_dim])
        self.__create_image(epoch=epoch, dim=dim, save_path=self.full_save_path_random_images, noise=noise,
                            target_image_size=target_image_size)

        # RangeImages: only works if latent_dim is 2
        if self.latent_dim == 2:
            noise = np.mgrid[-2:2.1:0.4, -2:2.1:0.4].reshape(2, -1).T
            dim = (11, 11)
            self.__create_image(epoch=epoch, dim=dim, save_path=self.full_save_path_range_images, noise=noise,
                                target_image_size=target_image_size)

    def __create_image(self, epoch, save_path, noise, target_image_size, dim):
        """
        Plots an image matrix by given noise and saves it to file if predefined.
        Matrix size depends on dim.
        :param epoch: Current epoch in the fit process.
        :param save_path: Save path for either random or range images.
        :param noise: Noise to get generator image data.
        :param target_image_size: Desired target image size.
        :param dim: Dimension of the plotted matrix, which is the number of dim[0] * dim[1] images.
        """
        examples = dim[0] * dim[1]

        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(examples, target_image_size, target_image_size, 1)
        generated_images = ((generated_images + 1) * target_image_size).astype("int")

        plt.figure(figsize=dim)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
        plt.title(f"Epoch {epoch}")
        plt.tight_layout()

        if self.save_images:
            epoch_zeros = str(epoch).zfill(5)
            plt.savefig(f"{save_path}Epoch_{epoch_zeros}.png")

        plt.show()
        plt.close()

    def fit_gan(self, epochs, batch_size):
        """
        Fits the GAN for a number of epochs. Model and images will be shown/saved as predefined.
        :param epochs: Epochs for training process.
        :param batch_size: Batch size for training process.
        :return: Returns the GAN-history.
        """
        batch_count = self.all_images_normalized.shape[0] / batch_size

        for e in range(1, epochs + 1):
            print('-' * 15, 'Epoche %d' % e, '-' * 15)

            for _ in tqdm(range(int(batch_count))):
                np.random.seed(None)
                real_images = self.all_images_normalized[np.random.randint(0,
                                                                           self.all_images_normalized.shape[0],
                                                                           size=batch_size)]

                z = np.random.normal(0, 1, size=[batch_size, self.latent_dim])
                fake_images = self.generator.predict(z)

                x_discriminator = np.concatenate([real_images, fake_images])
                y_discriminator = np.zeros(2 * batch_size)
                y_discriminator[batch_size:] = 1

                self.discriminator.trainable = True
                discriminator_history = self.discriminator.fit(x_discriminator, y_discriminator, verbose=False,
                                                               batch_size=batch_size)
                self.discriminator.trainable = False

                y_gan = np.zeros(batch_size)
                gan_history = self.gan.fit(z, y_gan, verbose=False, batch_size=batch_size)

            if e % 100 == 0:
                print("\n\ndiscriminator fit results:")
                print(discriminator_history.history)

                print("\ngan fit results:")
                print(gan_history.history)

            if self.save_model and e % self.save_model_every_epoch == 0:
                e_zeros = str(e).zfill(5)
                self.generator.save(f"{self.full_save_path}PokeGAN_{self.gantype}_Epoch_{e_zeros}.h5")

            if e % self.show_image_every_epoch == 0 or e % self.save_image_every_epoch == 0:
                self.__create_images(epoch=e)

        return gan_history

    def create_gifs(self, gif_from_random_images=True, gif_from_range_images=True, duration=900):
        """
        Creates gifs from all images in a folder depending on the save paths.
        :param gif_from_random_images: if True, a gif from all images in the random_Images folder will be created.
        :param gif_from_range_images: if True, a gif from all images in the range_Images folder will be created.
        :param duration: For how long a single image will be shown in the gif.
        """
        if gif_from_random_images:
            image_path = f"{self.full_save_path_random_images}Epoch_*.png"
            self.__create_gif(image_path, image_type='random_images', duration=duration)
        if gif_from_range_images:
            image_path = f"{self.full_save_path_range_images}Epoch_*.png"
            self.__create_gif(image_path, image_type='range_images', duration=duration)

    def __create_gif(self, image_path, image_type, duration):
        images, *imgs = [Image.open(f) for f in sorted(glob.glob(image_path))]
        images.save(fp=f"{self.full_save_path}{image_type}.gif",
                    format='GIF',
                    append_images=imgs,
                    save_all=True,
                    duration=duration,
                    loop=0)
