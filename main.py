import numpy as np
import tensorflow as tf
import datetime
import os
import show_hidden_outputs
import animatplot as amp
import matplotlib.pyplot as plt
import glob
from PIL import Image

class Dataset:
    def __init__(self, num_labeled):
        self.num_labeled = num_labeled

        true_file = glob.glob("Dataset/casting_data/train/ok_front/*.jpeg")
        false_file = glob.glob("Dataset/casting_data/train/def_front/*.jpeg")

        input_true = [np.array(Image.open(load_dir).resize((256,256), Image.NEAREST )) for load_dir in true_file[:500]]
        input_false = [np.array(Image.open(load_dir).resize((256,256), Image.NEAREST )) for load_dir in false_file[:500]]

        y_true = np.zeros(len(input_true))
        y_false = np.ones(len(input_false))

        x_train = np.concatenate([input_true, input_false])
        y_train = np.concatenate([y_true, y_false])

        p = np.random.permutation(len(x_train))
        self.x_train = x_train[p]
        self.y_train = y_train[p]

        true_file = glob.glob("Dataset/casting_data/test/ok_front/*.jpeg")
        false_file = glob.glob("Dataset/casting_data/test/def_front/*.jpeg")

        input_true = [np.array(Image.open(load_dir)) for load_dir in true_file]
        input_false = [np.array(Image.open(load_dir)) for load_dir in false_file]

        y_true = np.zeros(len(input_true))
        y_false = np.ones(len(input_false))

        x_test = np.concatenate([input_true, input_false])
        y_test = np.concatenate([y_true, y_false])

        p = np.random.permutation(len(x_test))
        self.x_test = x_test[p]
        self.y_test = y_test[p]

        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5
            return x

        def preprocess_labels(y):
            return y.reshape(-1,1)

        self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(self.y_train)

        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)

    def batch_labeled(self, batch_size):
        idx = np.random.randint(0,self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test


class SGAN():
    def __init__(self):
        img_rows = 256
        img_cols = 256
        channels = 3

        self.img_shape = (img_rows, img_cols, channels)

        self.z_dim = 10000
        self.num_classes = 10

        self.num_labeled = 100

        self.dataset = Dataset(self.num_labeled)

        self.discriminator_net = self.build_discriminator_net()

        self.discriminator_supervised = self.build_discriminator_supervised(self.discriminator_net)

        self.discriminator_unsupervised = self.build_discriminator_unsupervised(self.discriminator_net)

        self.acc = tf.metrics.CategoricalAccuracy()

        self.generator = self.build_generator()

        self.bc = tf.losses.BinaryCrossentropy()
        self.cc = tf.losses.CategoricalCrossentropy()
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.discriminator_un_r_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.discriminator_un_f_optimizer = tf.keras.optimizers.Adam(0.0001)


    def build_generator(self): # 生成器

        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.z_dim,)),
            tf.keras.layers.Dense(256 * 8 * 8),# 全結合
            tf.keras.layers.Reshape((8,8,256)),# 8*8*256のテンソルに変換
            tf.keras.layers.Conv2DTranspose(128, kernel_size=3,strides=2, padding="same"),# 転置畳み込み層により、8*8*256を16*16*128のテンソルに変換

            tf.keras.layers.BatchNormalization(),# バッチ正規化
            tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化

            tf.keras.layers.Conv2DTranspose(64, kernel_size=3,strides=2, padding="same"),# 転置畳み込み層により、16*16*128を32*32*64のテンソルに変換

            tf.keras.layers.BatchNormalization(),# バッチ正規化
            tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化

            tf.keras.layers.Conv2DTranspose(32,kernel_size=3,strides=2, padding="same"),# 転置畳み込み層により32*32*64を64*64*32のテンソルに変換

            tf.keras.layers.BatchNormalization(),# バッチ正規化
            tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化

            tf.keras.layers.Conv2DTranspose(16,kernel_size=3,strides=2, padding="same"),# 転置畳み込み層により64*64*32を128*128*16のテンソルに変換

            tf.keras.layers.BatchNormalization(),# バッチ正規化
            tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化

            tf.keras.layers.Conv2DTranspose(3,kernel_size=3,strides=2,padding="same"),# 転置畳み込み層により128*128*32を256*256*3のテンソルに変換

            tf.keras.layers.Activation("tanh") # tanh関数を用いた出力層 256*256*3
        ])

        return model

    def build_discriminator_net(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16,kernel_size=3 ,strides=2, input_shape=self.img_shape, padding="same"), # 256*256*3を128*128*16のテンソルにするたたみ込み層
            tf.keras.layers.LeakyReLU(alpha=0.01),# LeakyReLUによる活性化
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),# 128*128*16を64*64*32のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),# 64*64*32を32*32*64のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),# 32*32*64を16*16*128のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),# 16*16*128を8*8*256のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),# 8*8*256を4*4*512のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes)
        ])

        return model

    def build_discriminator_supervised(self, discriminator_net):
        model = tf.keras.models.Sequential([
            discriminator_net,
            tf.keras.layers.Activation('softmax')
        ])

        return model

    def build_discriminator_unsupervised(self, discriminator_net):
        def predict(x):
            prediction = 1.0 - (1.0 / (tf.keras.backend.sum(tf.keras.backend.exp(x), axis=-1,keepdims=True) + 1.0))
            return prediction

        model = tf.keras.Sequential([
            discriminator_net,
            tf.keras.layers.Lambda(predict)
        ])

        return model


    def discriminator_supervised_loss(self, y_true, y_pred):
        return self.cc(y_true, y_pred)

    def d_unsupervised_loss_real(self, real_output):
        real_loss = self.bc(tf.ones_like(real_output), real_output)
        return real_loss

    def d_unsupervised_loss_fake(self, fake_output):
        fake_loss = self.bc(tf.zeros_like(fake_output), fake_output)
        return fake_loss

    def discriminator_unsupervised_loss(self, real_loss, fake_loss):
        total_loss = (real_loss + fake_loss)
        return total_loss

    def generator_loss(self, fake_output):
        return self.bc(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, imgs, labels, imgs_unlabeled, batch_size):
        z = tf.random.normal([batch_size, self.z_dim])

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_un_r_tape, tf.GradientTape() as disc_un_f_tape:
            labels_pred = self.discriminator_supervised(imgs,training=True)
            d_loss_supervised = self.discriminator_supervised_loss(labels, labels_pred)

            gen_imgs = self.generator(z,training=True)

            real_output = self.discriminator_unsupervised(imgs_unlabeled,training=True)
            fake_output = self.discriminator_unsupervised(gen_imgs,training=True)
            d_loss_real = self.d_unsupervised_loss_real(real_output)
            d_loss_fake = self.d_unsupervised_loss_fake(fake_output)
            d_loss_unsupervised = self.discriminator_unsupervised_loss(d_loss_real, d_loss_fake)

            g_loss = self.generator_loss(fake_output)

        self.acc.update_state(labels, labels_pred)
        gradients_of_discriminator = disc_tape.gradient(d_loss_supervised, self.discriminator_supervised.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator_un_r = disc_un_r_tape.gradient(d_loss_real, self.discriminator_unsupervised.trainable_variables)
        gradients_of_discriminator_un_f = disc_un_f_tape.gradient(d_loss_fake, self.discriminator_unsupervised.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_supervised.trainable_variables))
        self.discriminator_un_r_optimizer.apply_gradients(zip(gradients_of_discriminator_un_r, self.discriminator_unsupervised.trainable_variables))
        self.discriminator_un_f_optimizer.apply_gradients(zip(gradients_of_discriminator_un_f, self.discriminator_unsupervised.trainable_variables))

        return g_loss, d_loss_supervised, d_loss_unsupervised

    def train(self, iterations,batch_size,sample_interval):
        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        log_dir = "./logs/SGAN/" + now.strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

        imgs_unlabeled = self.dataset.batch_unlabeled(batch_size)
        #tf.summary.trace_on(graph=True, profiler=True)
        for iteration in range(iterations):
            # 識別器の訓練

            imgs, labels = self.dataset.batch_labeled(batch_size)
            labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
            imgs_unlabeled = self.dataset.batch_unlabeled(batch_size)

            g_loss, d_loss_supervised, d_loss_unsupervised = self.train_step(tf.constant(imgs), tf.constant(labels), tf.constant(imgs_unlabeled), tf.constant(batch_size))
            accuracy = self.acc.result()

            if (iteration + 1) % sample_interval == 0:
                # 訓練の進捗を出力する
                print("%d [D loss supervised: %f, acc.: %.2f%%] [D loss unsupervised: %f] [G loss: %f]" % (iteration + 1, d_loss_supervised, 100.0 * accuracy, d_loss_unsupervised, g_loss))

                # 訓練終了後に図示するために、損失と精度を保存する
                with self.summary_writer.as_default():
                    tf.summary.scalar("d_loss_supervised", d_loss_supervised,iteration + 1)
                    tf.summary.scalar("d_loss_unsupervised", d_loss_unsupervised,iteration + 1)
                    tf.summary.scalar("g_loss", g_loss,iteration + 1)
                    tf.summary.scalar("accuracy", 100.0 * accuracy,iteration + 1)
                self.sample_images(iteration + 1)
                self.acc.reset_states()

        #with self.summary_writer.as_default():
        #    tf.summary.trace_export(name="SGAN",step=0,profiler_outdir=log_dir)

    def sample_images(self, step, image_grid_rows=4, image_grid_columns=4):

        # ランダムノイズのサンプリング
        z = tf.random.normal([16, self.z_dim])

        # ランダムノイズを使って画像を生成
        gen_imgs = self.generator.predict(z)

        # 画像の画素値を[0, 1]の範囲にスケーリング
        gen_imgs = 0.5 * gen_imgs + 0.5

        # 以下matplot処理
        for i in range(image_grid_rows * image_grid_columns):
            name = 'img_' + str(i)
            with self.summary_writer.as_default():
                tf.summary.image(name, tf.reshape(gen_imgs[i,:,:,:], [-1,256,256,3]), step=step, max_outputs=1)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        for k in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[k], True)

    iterations = 30000
    batch_size = 32
    sample_interval = 1000

    sgan = SGAN()
    sgan.train(iterations,batch_size, sample_interval)
    #sgan.generator.summary()
    #sgan.discriminator_supervised.layers[0].summary()
