import numpy as np
import tensorflow as tf
import datetime
import pytz

class GAN:
    def __init__(self):
        img_rows = 28
        img_cols = 28
        channels = 1

        self.img_shape = (img_rows, img_cols, channels)

        self.z_dim = 100

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.bc = tf.losses.BinaryCrossentropy()

        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam()

    def build_generator(self): # 生成器

        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.z_dim,)),
            tf.keras.layers.Dense(128),# 全結合
            tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化
            tf.keras.layers.Dense(28 * 28 * 1, activation="tanh"), # tanh関数を用いた出力層
            tf.keras.layers.Reshape(self.img_shape) # 生成器の出力を画像サイズに合わせる
        ])

        return model

    def build_discriminator(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),# 入力画像を1列に
            tf.keras.layers.Dense(128),# 全結合
            tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化
            tf.keras.layers.Dense(1,activation="sigmoid") # sigmoid関数を通して出力
        ])

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.bc(tf.ones_like(real_output), real_output)
        fake_loss = self.bc(tf.zeros_like(fake_output), fake_output)
        total_loss = (real_loss + fake_loss)
        return total_loss

    def generator_loss(self, fake_output):
        return self.bc(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, imgs, batch_size):
        z = tf.random.normal([batch_size, self.z_dim])

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            gen_imgs = self.generator(z,training=True)

            real_output = self.discriminator(imgs,training=True)
            fake_output = self.discriminator(gen_imgs,training=True)

            d_loss = self.discriminator_loss(real_output, fake_output)
            g_loss = self.generator_loss(fake_output)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return g_loss, d_loss

    def train(self, iterations,batch_size,sample_interval):
        log_dir = "./logs/gan/" + datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

        (X_train,_),(_,_) = tf.keras.datasets.mnist.load_data()

        X_train = X_train / 127.5 - 1.0 # [0,255]の範囲の画素値を[-1,1]にスケーリング
        X_train = np.expand_dims(X_train,axis=3)

        #tf.summary.trace_on(graph=True, profiler=True)
        for iteration in range(iterations):

            # 本物の画像をランダムに取り出したバッチの作成
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]

            # 訓練
            g_loss, d_loss = self.train_step(tf.constant(imgs), tf.constant(batch_size))

            if (iteration + 1) % sample_interval == 0:
                # 訓練の進捗を出力する
                print("%d [D loss: %f] [G loss: %f]" % (iteration + 1, d_loss, g_loss))

                # 訓練終了後に図示するために、損失と精度を保存する
                with self.summary_writer.as_default():
                    tf.summary.scalar("d_loss", d_loss,iteration + 1)
                    tf.summary.scalar("g_loss", g_loss,iteration + 1)

                # サンプル画像を生成し出力
                self.sample_images(iteration + 1)

        #with self.summary_writer.as_default():
        #    tf.summary.trace_export(name="gan",step=0,profiler_outdir=log_dir)


    def sample_images(self, step, image_grid_rows=4, image_grid_columns=4):
        # ランダムノイズのサンプリング
        z = tf.random.normal([image_grid_rows * image_grid_columns, self.z_dim])

        # ランダムノイズを使って画像を生成
        gen_imgs = self.generator.predict(z)

        # 画像の画素値を[0, 1]の範囲にスケーリング
        gen_imgs = 0.5 * gen_imgs + 0.5

        # 以下matplot処理
        for i in range(image_grid_rows * image_grid_columns):
            name = 'img_' + str(i)
            with self.summary_writer.as_default():
                tf.summary.image(name, tf.reshape(gen_imgs[i,:,:,0], [-1,28,28,1]), step=step, max_outputs=1)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        for k in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[k], True)

    iterations = 200000
    batch_size = 128
    sample_interval = 1000

    gan = GAN()
    gan.train(iterations, batch_size, sample_interval)
