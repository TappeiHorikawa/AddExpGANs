import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def img_marge(img,x,y):
    hcon_images = []
    for i in range(y):
        hcon_images.append(np.concatenate(img[x * i :x * (i+1),:,:], axis=1))
    return np.concatenate(hcon_images, axis=0)

def show_hidden_outputs(model,input_img):
    # model = sgan.discriminator_supervised.layers[0]

    names = [l.name for l in model.layers]
    #print(names)


    for name in names:
        print(name)
        hidden_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name).output)

        hidden_output = hidden_layer_model.predict(input_img[0:1,:,:,:])
        w = hidden_output

        w_scale = ((w - w.min()) / (w.max() - w.min()) * 255).astype('uint8')
        #print(w_scale.min(), w_scale.max())

        if (len(w_scale.shape) == 4):
            w_transpose = w_scale.transpose(3, 0, 1, 2)
            #print(w_transpose.shape)
            w_transpose = np.squeeze(w_transpose)
            print(w_transpose.shape)
            print(28//w_transpose.shape[1])
            img = img_marge(w_transpose,16,w_transpose.shape[0]//16)
            img = img.repeat(28//w_transpose.shape[1], axis=0).repeat(28//w_transpose.shape[2], axis=1)
            plt.figure(dpi=600)
            plt.imshow(img, cmap = "gray")
            plt.show()

        elif (128 <= w_scale.shape[1]):
            w_transpose = np.squeeze(w_scale)
            w_transpose = w_transpose.reshape(w_transpose.shape[0]//128, 128)
            plt.figure(dpi=600)
            plt.imshow(w_transpose, cmap = "gray")
            plt.show()
        else:
            plt.figure(dpi=600)
            plt.imshow(w_scale, cmap = "gray")
            plt.show()

def marge_hidden_outputs(model,input_img, summary_writer = None, step = 0):
    # model = sgan.discriminator_supervised.layers[0]

    names = [l.name for l in model.layers]
    #print(names)


    for name in names:
        print(name)
        hidden_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name).output)

        hidden_output = hidden_layer_model.predict(input_img[0:1,:,:,:])
        w = hidden_output

        w_scale = ((w - w.min()) / (w.max() - w.min()) * 255).astype('uint8')
        #print(w_scale.min(), w_scale.max())

        if (len(w_scale.shape) == 4):
            w_transpose = w_scale.transpose(3, 0, 1, 2)
            #print(w_transpose.shape)
            w_transpose = np.squeeze(w_transpose)

            out_img = 0
            for img in w_transpose:
                out_img = 0.5 * out_img +  0.5 * img
            img = np.squeeze(out_img)
            print(w_transpose.shape)
            print(28//w_transpose.shape[1])
            img = img.repeat(28//w_transpose.shape[1], axis=0).repeat(28//w_transpose.shape[2], axis=1)
            inp_img = np.squeeze(input_img[0:1,:,:,:])
            cmap = ""
            summary_out = 0.9 * inp_img + 0.1 * img

        elif (128 <= w_scale.shape[1]):
            w_transpose = np.squeeze(w_scale)
            w_transpose = w_transpose.reshape(w_transpose.shape[0]//128, 128)
            cmap = "gray"
            summary_out = w_scale

        else:
            cmap = "gray"
            summary_out = w_scale

        if summary_writer != None:
            summary_out = tf.expand_dims(summary_out, 0)
            summary_out = tf.expand_dims(summary_out, -1)
            with summary_writer.as_default():
                tf.summary.image(name, summary_out, step=step, max_outputs=1)
        else:
            plt.imshow(summary_out, cmap = cmap)
            plt.show()

def marge_hidden_outputs_generator(model, input_img, summary_writer = None, step = 0):
    # model = sgan.discriminator_supervised.layers[0]

    names = [l.name for l in model.layers]
    #print(names)


    for name in names:
        print(name)
        hidden_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name).output)

        hidden_output = hidden_layer_model.predict(input_img)
        w = hidden_output

        w_scale = ((w - w.min()) / (w.max() - w.min()) * 255).astype('uint8')
        #print(w_scale.min(), w_scale.max())

        if (len(w_scale.shape) == 4):
            w_transpose = w_scale.transpose(3, 0, 1, 2)
            w_transpose = np.squeeze(w_transpose)

            print(w_transpose.shape)
            print(28//w_transpose.shape[1])
            if (len(w_transpose.shape) != 2):
                out_img = 0
                for img in w_transpose:
                    out_img = 0.5 * out_img +  0.5 * img
                img = np.squeeze(out_img)

                img = img.repeat(28//w_transpose.shape[1], axis=0).repeat(28//w_transpose.shape[2], axis=1)
                cmap = ""
                summary_out = 1.0 * img

            else:
                cmap = ""
                summary_out = w_transpose

        elif (128 <= w_scale.shape[1]):
            w_transpose = np.squeeze(w_scale)
            w_transpose = w_transpose.reshape(w_transpose.shape[0]//128, 128)
            cmap = "gray"
            summary_out = w_transpose
        else:
            cmap = "gray"
            summary_out = w_scale

        if summary_writer != None:
            summary_out = tf.expand_dims(summary_out, 0)
            summary_out = tf.expand_dims(summary_out, -1)
            with summary_writer.as_default():
                tf.summary.image(name, summary_out, step=step, max_outputs=1)
        else:
            plt.imshow(summary_out, cmap = cmap)
            plt.show()

def marge_hidden_outputs_gif(model,input_img):
    # model = sgan.discriminator_supervised.layers[0]

    names = [l.name for l in model.layers]
    #print(names)
    out_list = []

    for name in names:
        #print(name)
        hidden_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name).output)

        hidden_output = hidden_layer_model.predict(input_img[0:1,:,:,:])
        w = hidden_output

        w_scale = ((w - w.min()) / (w.max() - w.min()) * 255).astype('uint8')
        #print(w_scale.min(), w_scale.max())

        if (len(w_scale.shape) == 4):
            w_transpose = w_scale.transpose(3, 0, 1, 2)
            #print(w_transpose.shape)
            w_transpose = np.squeeze(w_transpose)

            out_img = 0
            for img in w_transpose:
                out_img = 0.5 * out_img +  0.5 * img
            img = np.squeeze(out_img)
            #print(w_transpose.shape)
            #print(28//w_transpose.shape[1])
            img = img.repeat(28//w_transpose.shape[1], axis=0).repeat(28//w_transpose.shape[2], axis=1)
            inp_img = np.squeeze(input_img[0:1,:,:,:])
            summary_out = 0.9 * inp_img + 0.1 * img

        elif (128 <= w_scale.shape[1]):
            w_transpose = np.squeeze(w_scale)
            w_transpose = w_transpose.reshape(w_transpose.shape[0]//128, 128)
            summary_out = w_scale

        else:
            summary_out = w_scale

        out_list.append(summary_out)
    return out_list

def marge_hidden_outputs_generator_gif(model, input_img):
    # model = sgan.discriminator_supervised.layers[0]

    names = [l.name for l in model.layers]
    #print(names)
    out_list = []

    for name in names:
        #print(name)
        hidden_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name).output)

        hidden_output = hidden_layer_model.predict(input_img)
        w = hidden_output

        w_scale = ((w - w.min()) / (w.max() - w.min()) * 255).astype('uint8')
        #print(w_scale.min(), w_scale.max())

        if (len(w_scale.shape) == 4):
            w_transpose = w_scale.transpose(3, 0, 1, 2)
            w_transpose = np.squeeze(w_transpose)

            #print(w_transpose.shape)
            #print(28//w_transpose.shape[1])
            if (len(w_transpose.shape) != 2):
                out_img = 0
                for img in w_transpose:
                    out_img = 0.5 * out_img +  0.5 * img
                img = np.squeeze(out_img)

                img = img.repeat(28//w_transpose.shape[1], axis=0).repeat(28//w_transpose.shape[2], axis=1)
                summary_out = 1.0 * img

            else:
                summary_out = w_transpose

        elif (128 <= w_scale.shape[1]):
            w_transpose = np.squeeze(w_scale)
            w_transpose = w_transpose.reshape(w_transpose.shape[0]//128, 128)
            summary_out = w_transpose
        else:
            summary_out = w_scale

        out_list.append(summary_out)
    return out_list
