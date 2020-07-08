import numpy as np
import tensorflow as tf
import datetime

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)


img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100

def build_generator(z_dim): # 生成器

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256 * 7 * 7),# 全結合
        tf.keras.layers.Reshape((7,7,256)),# 7*7*256のテンソルに変換

        tf.keras.layers.Conv2DTranspose(128, kernel_size=3,strides=2, padding="same"),# 転置畳み込み層により、7*7*256を14*14*128のテンソルに変換

        tf.keras.layers.BatchNormalization(),# バッチ正規化
        tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化

        tf.keras.layers.Conv2DTranspose(64,kernel_size=3,strides=1, padding="same"),# 転置畳み込み層により14*14*128を14*14*64のテンソルに変換

        tf.keras.layers.BatchNormalization(),# バッチ正規化
        tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化

        tf.keras.layers.Conv2DTranspose(1,kernel_size=3,strides=2,padding="same"),# 転置畳み込み層により14*14*64を28*28*1のテンソルに変換

        tf.keras.layers.Activation("tanh") # tanh関数を用いた出力層
    ])

    return model

def build_discriminator(img_shape): # 識別器

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32,kernel_size=3 ,strides=2, input_shape=img_shape, padding="same"), # 28*28*1を14*14*32のテンソルにするたたみ込み層
        tf.keras.layers.LeakyReLU(alpha=0.01),# LeakyReLUによる活性化

        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"),# 14*14*32を7*7*64のテンソルにするたたみ込み層
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),

        tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding="same"),# 7*7*64を3*3*128のテンソルにするたたみ込み層
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation="sigmoid") # sigmoid関数を通して出力
    ])

    return model

discriminator = build_discriminator(img_shape)
generator = build_generator(z_dim)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output): # 識別器誤差
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output): # 生成器誤差
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# オプティマイザはネットワークごとに必要
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

log_dir = "./logs/dcgan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(logdir=log_dir)

@tf.function
def train_step(images, iteration, sample_interval):
    noise = tf.random.normal([128, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss



def train(iterations, sample_interval):
    (X_train,_),(_,_) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_train = X_train / 127.5 - 1.0 # [0,255]の範囲の画素値を[-1,1]にスケーリング

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(60000).batch(128, drop_remainder=True)

    for iteration in range(iterations):

        for image_batch in train_dataset:
            disc_loss, gen_loss = train_step(image_batch, tf.constant(iteration, dtype=tf.int64), tf.constant(sample_interval, dtype=tf.int64))
        if (iteration + 1) % sample_interval == 0:
        # 訓練終了後に図示するために、損失と精度を保存する
            with summary_writer.as_default():
                tf.summary.scalar("d_loss", disc_loss,iteration + 1)
                tf.summary.scalar("g_loss", gen_loss,iteration + 1)


            # 訓練の進捗を出力する
            tf.print(iteration + 1, " [D loss: ", disc_loss, "%] [G loss: ", gen_loss, "]", sep='')

            # サンプル画像を生成し出力
            sample_images(generator, iteration + 1)

def sample_images(generator, step, image_grid_rows=4, image_grid_columns=4):

    # ランダムノイズのサンプリング
    z = tf.random.normal([16, 100])

    # ランダムノイズを使って画像を生成
    gen_imgs = generator(z)

    # 画像の画素値を[0, 1]の範囲にスケーリング
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 以下matplot処理
    for i in range(image_grid_rows * image_grid_columns):
        name = 'img_' + str(i)
        with summary_writer.as_default():
            tf.summary.image(name, tf.reshape(gen_imgs[i,:,:,0], [-1,28,28,1]), step=step, max_outputs=1)

iterations = 100
batch_size = 128
sample_interval = 5

train(iterations,sample_interval)
