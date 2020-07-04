import numpy as np
import tensorflow as tf
import datetime

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

def build_discriminator(img_shape):

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

def build_gan(generator, discriminator):
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss="binary_crossentropy",optimizer="Adam",metrics=["accuracy"])

generator = build_generator(z_dim)

discriminator.trainable = False # 生成器の構築中は識別器のパラメータを固定

gan = build_gan(generator,discriminator) # 生成器の訓練のため、識別器は固定しGANモデルの構築とコンパイルをおこなう。
gan.compile(loss="binary_crossentropy", optimizer="Adam")

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(logdir=log_dir)

def train(iterations,batch_size,sample_interval):
    (X_train,_),(_,_) = tf.keras.datasets.mnist.load_data()

    X_train = X_train / 127.5 - 1.0 # [0,255]の範囲の画素値を[-1,1]にスケーリング
    X_train = np.expand_dims(X_train,axis=3)

    real = np.ones((batch_size,1)) # 本物の画像ラベルは1

    fake = np.zeros((batch_size,1)) # 偽物の画像ラベルは0

    for iteration in range(iterations):
        # 識別器の訓練

        # 本物の画像をランダムに取り出したバッチの作成
        idx = np.random.randint(0,X_train.shape[0],batch_size)
        imgs = X_train[idx]

        # 偽画像のバッチを作成
        z = np.random.normal(0,1,(batch_size, 100))
        gen_imgs = generator.predict(z)

        # 識別器の訓練
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real,d_loss_fake)

        # 生成器の訓練
        g_loss = gan.train_on_batch(z,real)

        if (iteration + 1) % sample_interval == 0:
            # 訓練終了後に図示するために、損失と精度を保存する
            with summary_writer.as_default():
                tf.summary.scalar("d_loss", d_loss,iteration + 1)
                tf.summary.scalar("g_loss", g_loss,iteration + 1)
                tf.summary.scalar("accuracy", 100.0 * accuracy,iteration + 1)

            # 訓練の進捗を出力する
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # サンプル画像を生成し出力
            sample_images(generator, iteration + 1)

def sample_images(generator, step, image_grid_rows=4, image_grid_columns=4):

    # ランダムノイズのサンプリング
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ランダムノイズを使って画像を生成
    gen_imgs = generator.predict(z)

    # 画像の画素値を[0, 1]の範囲にスケーリング
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 以下matplot処理
    for i in range(image_grid_rows * image_grid_columns):
        name = 'img_' + str(i)
        with summary_writer.as_default():
            tf.summary.image(name, tf.reshape(gen_imgs[i,:,:,0], [-1,28,28,1]), step=step, max_outputs=1)

iterations = 20000
batch_size = 128
sample_interval = 1000

train(iterations, batch_size, sample_interval)
