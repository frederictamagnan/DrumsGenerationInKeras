import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector,GRU,Bidirectional,CuDNNGRU,BatchNormalization,Reshape,Concatenate
from keras.layers.core import Flatten, Dense, Dropout, Lambda,Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras.models import load_model
from keras.utils import plot_model

import tensorflowjs as tfjs
batch_size=256
SEQ_LEN=96
INPUT_DIM=9
LINEAR_HIDDEN_SIZE=[64,32]
GRU_HIDDEN_SIZE=32
NUM_DIRECTIONS=2
BEAT=48

def create_gru_vae(seq_len=SEQ_LEN,
                   input_dim=INPUT_DIM,
                   gru_hidden_size=GRU_HIDDEN_SIZE,
                   linear_hidden_size=LINEAR_HIDDEN_SIZE,
                   num_directions=NUM_DIRECTIONS,
                   epsilon_std=1.,
                   beat=BEAT):




    #encpder layers
    simple_gru = GRU(gru_hidden_size)
    gru = Bidirectional(simple_gru)
    gru_out_dim = seq_len * gru_hidden_size * num_directions
    gru_in_dim = seq_len * input_dim

    bn0 = BatchNormalization()
    linear0 = Dense(linear_hidden_size[0], input_dim=gru_out_dim)
    bn1 = BatchNormalization()

    #decoder layers

    linear0d = Dense(gru_in_dim)
    bn0d = BatchNormalization()
    bn1d = BatchNormalization()

    linear1d = Dense(input_dim)
    bn2d = BatchNormalization()

    grud = GRU(input_dim, return_sequences=True,return_state=True)






    #encoder_part

    # first=Input(batch_shape=(batch_size,seq_len, input_dim))
    first=Input(shape=(seq_len,input_dim))
    x = gru(first)
    x = bn0(x)
    x = linear0(x)
    x = Activation('tanh')(x)
    h = bn1(x)

    # VAE Z layer
    z_mean = Dense(linear_hidden_size[1])(h)
    z_log_sigma = Dense(linear_hidden_size[1])(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(linear_hidden_size[1],),
                                  mean=0., stddev=epsilon_std)


        return z_mean + z_log_sigma * epsilon

    z = Lambda(sampling, output_shape=(linear_hidden_size[1],))([z_mean, z_log_sigma])
    z_ = Input(shape=(linear_hidden_size[1],))


    def decoder(z):
        n_sections = SEQ_LEN // beat
        b = beat
        list_tensor=[]
        x = linear0d(z)
        x = bn0d(x)
        x = Activation('tanh')(x)
        x = Reshape((seq_len, input_dim))(x)


        #custom sections

        for i in range(n_sections):

            if i==0:
                x, hn = grud(x)
            else:
                x,hn=grud(x)



            x = bn1d(x)
            print(x)
            x=Lambda(lambda x : x[:,:b,:])(x)
            print(x)
            x=linear1d(x)
            x=bn2d(x)
            x=Activation('sigmoid')(x)
            list_tensor.append(x)

        melody=Concatenate(axis=1)(list_tensor)
        last = Activation('tanh')(melody)

        return last

    last=decoder(z)
    last_g=decoder(z_)
    print(last)
    vae=Model(first,last)
    encoder=Model(first,z_mean)



    generator=Model(z_,last_g)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss


    vae.compile(optimizer='rmsprop', loss=vae_loss)

    generator.compile(optimizer='rmsprop', loss=vae_loss)


    return vae,encoder,generator

import numpy as np
X=np.load('/home/ftamagna/Documents/_AcademiaSinica/code/DrumFillsNI/data/train_x_drum_reduced_Pop.npy')

X=X[:1024]

model,_,generator=create_gru_vae()
plot_model(model, to_file='model.png')
model.fit(X,X,epochs=4)
model.save_weights('my_model_weights.h5')
print("DONE")
model.load_weights('my_model_weights.h5', by_name=True)
print("DONE")
generator.load_weights('my_model_weights.h5',by_name=True)
print("DONE")
# layer_dict = dict([(layer.name, layer) for layer in model.layers])
# layer_dict_g = dict([(layer.name, layer) for layer in generator.layers])
#
# for key in layer_dict_g.keys():
#     weights = layer_dict[key].get_weights()

tfjs.converters.save_keras_model(model, "./")
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, (1,32))
print(s.shape)

lol=generator.predict(s)

