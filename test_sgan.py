import numpy as np
import tensorflow as tf
import datetime
from copy import deepcopy
import os
#os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"

tf.random.set_seed(1234)
np.random.seed(1234)
np.set_printoptions(threshold=np.inf)
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[1:], 'GPU')
BUFFER_SIZE = 60000
BATCH_SIZE = 32
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_labels = train_labels.reshape(-1,1)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
#train_labeled_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_unlabeled_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_images)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

class Dataset:
    def __init__(self, num_labeled):
        self.num_labeled = num_labeled

        (x_train, y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5

            x = np.expand_dims(x,axis=3)

            return x

        def preprocess_labels(y):
            return y.reshape(-1,1)

        self.x_train = preprocess_imgs(x_train)
        self.y_train = preprocess_labels(y_train)

        self.x_test = preprocess_imgs(x_test)
        self.y_test = preprocess_labels(y_test)

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



iterations = 20000
batch_size = 32
sample_interval = 1000
z_dim = 100


img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)


num_classes = 10

num_labeled = 100

dataset = Dataset(num_labeled)

discriminator_net = build_discriminator_net(img_shape)

discriminator_supervised = build_discriminator_supervised(discriminator_net)
#discriminator_supervised.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="Adam")
#discriminator_supervised.compile(loss=tf.losses.CategoricalCrossentropy(), metrics=[tf.metrics.CategoricalAccuracy()], optimizer=tf.optimizers.Adam())

discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
#discriminator_unsupervised.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="Adam")
#discriminator_unsupervised.compile(loss=tf.losses.BinaryCrossentropy(), metrics=[tf.metrics.BinaryAccuracy()], optimizer=tf.optimizers.Adam())

generator = build_generator(z_dim)

#discriminator_unsupervised.trainable = False # 生成器の構築中は識別器のパラメータを固定

gan = build_gan(generator,discriminator_unsupervised) # 生成器の訓練のため、識別器は固定しGANモデルの構築とコンパイルをおこなう。
#gan.compile(loss="binary_crossentropy", optimizer="Adam")
#gan.compile(loss=tf.losses.BinaryCrossentropy(), optimizer=tf.optimizers.Adam())



bc = tf.losses.BinaryCrossentropy()
cc = tf.losses.CategoricalCrossentropy()

def discriminator_supervised_loss(y_true, y_pred):
    return cc(y_true, y_pred)+ 1e-12

def d_unsupervised_loss_real(real_output):
    real_loss = bc(tf.ones_like(real_output), real_output)
    return real_loss+ 1e-12

def d_unsupervised_loss_fake(fake_output):
    fake_loss = bc(tf.zeros_like(fake_output), fake_output)
    return fake_loss+ 1e-12

def discriminator_unsupervised_loss(real_loss, fake_loss):
    total_loss = (real_loss + fake_loss)
    return total_loss+ 1e-12

def generator_loss(fake_output):
    return bc(tf.ones_like(fake_output), fake_output)+ 1e-12



log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(logdir=log_dir)



"""
# 訓練前トレーニングデータ正解率
x, y = dataset.training_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
_, accuracy = discriminator_supervised.evaluate(x, y)
print("Before Training Accuracy: %.2f%%" % (100 * accuracy))

# 訓練前テストデータ正解率
x, y = dataset.test_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
_, accuracy = discriminator_supervised.evaluate(x,y)
print("Before Test Accuracy: %.2f%%" % (100 * accuracy))
"""
tf.random.set_seed(1234)
train(iterations, batch_size, sample_interval)

#discriminator_supervised.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="Adam")

# 訓練後トレーニングデータ正解率
acc1 = tf.metrics.CategoricalAccuracy()
x, y = dataset.training_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
y_ = discriminator_supervised(x)
acc1.update_state(y,y_)
print("Training Accuracy: %.2f%%" % (100 * acc1.result()))

# 訓練後テストデータ正解率
acc2 = tf.metrics.CategoricalAccuracy()
x, y = dataset.test_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
y_ = discriminator_supervised(x)
acc2.update_state(y,y_)
print("Test Accuracy: %.2f%%" % (100 * acc2.result()))



# Fully supervised classifier with the same network architecture as the SGAN Discriminator
mnist_classifier = build_discriminator_supervised(build_discriminator_net(img_shape))
mnist_classifier.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer="Adam")

# Fit前テスト正解率
x, y = dataset.test_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
_, accuracy = mnist_classifier.evaluate(x, y)
print("Fit Before Test Accuracy: %.2f%%" % (100 * accuracy))
iterations = 30
imgs, labels = dataset.training_set()
# One-hot encode labels
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
# Train the classifier
training = mnist_classifier.fit(x=imgs, y=labels, batch_size=batch_size, epochs=iterations, verbose=1)
losses = training.history['loss']
accuracies = training.history['accuracy']


# フィット後トレーニング正解率
x, y = dataset.training_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
_, accuracy = mnist_classifier.evaluate(x, y)
print("Training Accuracy: %.2f%%" % (100 * accuracy))

# フィット後テスト正解率
x, y = dataset.test_set()
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
_, accuracy = mnist_classifier.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))

