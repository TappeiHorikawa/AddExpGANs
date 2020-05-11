import numpy as np
import tensorflow as tf
import datetime

def build_generator(img_shape,z_dim): # 生成器

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128),# 全結合
        tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化
        tf.keras.layers.Dense(28 * 28 * 1, activation="tanh"), # tanh関数を用いた出力層
        tf.keras.layers.Reshape(img_shape) # 生成器の出力を画像サイズに合わせる
    ])

    return model

def build_discriminator(img_shape):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),# 入力画像を1列に
        tf.keras.layers.Dense(128),# 全結合
        tf.keras.layers.LeakyReLU(alpha=0.01), # LeakyReLUによる活性化
        tf.keras.layers.Dense(1,activation="sigmoid") # sigmoid関数を通して出力
    ])

    return model

def build_gan(generator, discriminator):
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])

    return model

def train(iterations,batch_size,sample_interval, summary_writer,generator, discriminator, gan):
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
            sample_images(generator, summary_writer, iteration + 1)

def sample_images(generator, summary_writer, step, image_grid_rows=4, image_grid_columns=4):
    z_dim = 100
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

def main():

    img_rows = 28
    img_cols = 28
    channels = 1

    img_shape = (img_rows, img_cols, channels)

    z_dim = 100

    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

    generator = build_generator(img_shape,z_dim)

    discriminator.trainable = False # 生成器の構築中は識別器のパラメータを固定

    gan = build_gan(generator,discriminator) # 生成器の訓練のため、識別器は固定しGANモデルの構築とコンパイルをおこなう。
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    iterations = 200000
    batch_size = 128
    sample_interval = 1000

    train(iterations, batch_size, sample_interval, summary_writer,generator, discriminator, gan)

if __name__ == "__main__":
    main()