"""
Created by Stefanie Müller, October 2021
Based on my deep learning course project from september 2021.
"""

from PokeGAN import PokeGAN

if __name__ == '__main__':
    gan = PokeGAN(gantype='CNN',  # 'Dense' or 'CNN'
                  n_all_images=34,
                  save_model=True,
                  save_images=True,
                  show_image_every_epoch=100,
                  save_image_every_epoch=500,
                  save_model_every_epoch=1000)

    # save_path, save_path_random, save_path_range = gan.get_save_paths()

    gan.load_data(datapath='PokeData/', target_image_size=128)

    gan.plot_images()

    gan.generator_create()
    gan.discriminator_create(dropout_rate=0.5)  # dropout_rate=0.5 for CNN and 0.2 for Dense
    gan.combine_gan()

    gan.fit_gan(epochs=15000, batch_size=17)

    gan.create_gifs()
