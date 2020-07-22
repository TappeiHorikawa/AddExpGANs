import numpy as np
import tensorflow as tf
import datetime
from keras import backend as K

tf.random.set_seed(1234)
np.random.seed(1234)
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[1:], 'GPU')

class Dataset:
    def __init__(self, num_labeled):
        self.num_labeled = num_labeled

        (self.x_train, self.y_train),(self.x_test,self.y_test) = tf.keras.datasets.mnist.load_data()

        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5

            x = np.expand_dims(x,axis=3)

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
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.z_dim = 100
        self.num_classes = 10

        self.num_labeled = 100

        self.dataset = Dataset(self.num_labeled)

        self.discriminator_net = self.build_discriminator_net(self.img_shape)

        self.discriminator_supervised = self.build_discriminator_supervised(self.discriminator_net)

        self.discriminator_unsupervised = self.build_discriminator_unsupervised(self.discriminator_net)

        self.acc = tf.metrics.CategoricalAccuracy()

        self.generator = self.build_generator(self.z_dim)

        self.bc = tf.losses.BinaryCrossentropy()
        self.cc = tf.losses.CategoricalCrossentropy()
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_un_r_optimizer = tf.keras.optimizers.Adam()
        self.discriminator_un_f_optimizer = tf.keras.optimizers.Adam()

        self.log_dir = "./logs/SGAN/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)



    def build_generator(self, z_dim): # 生成器

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

    def build_discriminator_net(self, img_shape):

        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(32,kernel_size=3 ,strides=2, input_shape=img_shape, padding="same"), # 28*28*1を14*14*32のテンソルにするたたみ込み層
            tf.keras.layers.LeakyReLU(alpha=0.01),# LeakyReLUによる活性化

            tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"),# 14*14*32を7*7*64のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding="same"),# 7*7*64を3*3*128のテンソルにするたたみ込み層
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_classes) # sigmoid関数を通して出力
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
            prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1,keepdims=True) + 1.0))
            return prediction

        model = tf.keras.Sequential([
            discriminator_net,
            tf.keras.layers.Lambda(predict)
        ])

        return model


    def discriminator_supervised_loss(self, y_true, y_pred):
        return self.cc(y_true, y_pred)+ 1e-12

    def d_unsupervised_loss_real(self, real_output):
        real_loss = self.bc(tf.ones_like(real_output), real_output)
        return real_loss+ 1e-12

    def d_unsupervised_loss_fake(self, fake_output):
        fake_loss = self.bc(tf.zeros_like(fake_output), fake_output)
        return fake_loss+ 1e-12

    def discriminator_unsupervised_loss(self, real_loss, fake_loss):
        total_loss = (real_loss + fake_loss)
        return total_loss+ 1e-12

    def generator_loss(self, fake_output):
        return self.bc(tf.ones_like(fake_output), fake_output)+ 1e-12

    @tf.function
    def train_step(self, imgs, labels, imgs_unlabeled):
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
        tf.random.set_seed(1234)

        tf.summary.trace_on(graph=True, profiler=True)
        for iteration in range(iterations):
            # 識別器の訓練

            imgs, labels = self.dataset.batch_labeled(batch_size)

            labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)

            imgs_unlabeled = self.dataset.batch_unlabeled(batch_size)

            g_loss, d_loss_supervised, d_loss_unsupervised = self.train_step(tf.constant(imgs), tf.constant(labels), tf.constant(imgs_unlabeled))
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

        with self.summary_writer.as_default():
            tf.summary.trace_export(name="SGAN",step=0,profiler_outdir=self.log_dir)

    def sample_images(self, step, image_grid_rows=4, image_grid_columns=4):

        # ランダムノイズのサンプリング
        z = tf.random.normal([16, 100])

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
    iterations = 4000
    batch_size = 256
    sample_interval = 800

    sgan = SGAN()
    sgan.train(iterations, batch_size, sample_interval)

    x, y = sgan.dataset.training_set()
    y = tf.keras.utils.to_categorical(y, num_classes=sgan.num_classes)

    # Compute classification accuracy on the training set
    _, accuracy = sgan.discriminator_supervised.evaluate(x, y)
    print("Training Accuracy: %.2f%%" % (100 * accuracy))

    x, y = sgan.dataset.test_set()
    y = tf.keras.utils.to_categorical(y, num_classes=sgan.num_classes)

    _, accuracy = sgan.discriminator_supervised.evaluate(x,y)
    print("Test Accuracy: %.2f%%" % (100 * accuracy))

    # Fully supervised classifier with the same network architecture as the SGAN Discriminator
    mnist_classifier = sgan.build_discriminator_supervised(sgan.build_discriminator_net(sgan.img_shape))
    mnist_classifier.compile(loss='categorical_crossentropy',
                             metrics=['accuracy'],
                             optimizer="Adam")

    imgs, labels = sgan.dataset.training_set()
    # One-hot encode labels
    labels = tf.keras.utils.to_categorical(labels, num_classes=sgan.num_classes)

    # Train the classifier
    training = mnist_classifier.fit(x=imgs,
                                    y=labels,
                                    batch_size=32,
                                    epochs=30,
                                    verbose=1)
    losses = training.history['loss']
    accuracies = training.history['accuracy']


    x, y = sgan.dataset.training_set()
    y = tf.keras.utils.to_categorical(y, num_classes=sgan.num_classes)

    # Compute classification accuracy on the training set
    _, accuracy = mnist_classifier.evaluate(x, y)
    print("Training Accuracy: %.2f%%" % (100 * accuracy))

    x, y = sgan.dataset.test_set()
    y = tf.keras.utils.to_categorical(y, num_classes=sgan.num_classes)

    # Compute classification accuracy on the test set
    _, accuracy = mnist_classifier.evaluate(x, y)
    print("Test Accuracy: %.2f%%" % (100 * accuracy))