# %%
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

# %% model 1 - 
def create_vae_model_1():
    encoder_input = tfkl.Input(shape=(None,))
    encoder_output, state_h, state_c = tfkl.LSTM(64, return_state=True, name="encoder")(
        encoder_input
    )
    encoder_state = [state_h, state_c]

    decoder_input = tfkl.Input(shape=(None,))
    decoder_embedded = tfkl.Embedding(input_dim=decoder_vocab, output_dim=64)(
        decoder_input
    )

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = tfkl.LSTM(64, name="decoder")(
        decoder_embedded, initial_state=encoder_state
    )
    return vae, encoder_output, decoder_output



# %% model 2 - https://www.tensorflow.org/probability/examples/Probabilistic_Layers_VAE
def create_vae_model_2():
    encoded_size = 16
    base_depth = 32
    input_shape = (500,1)

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                            reinterpreted_batch_ndims=1)

    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=input_shape),
        tfkl.LSTM(64),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
                activation=None),
        tfpl.MultivariateNormalTriL(
            encoded_size,
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
    ])

    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[encoded_size]),
        tfkl.LSTM(64, return_sequences=True),
        tfkl.Dense(100, activation='relu'),
        tfkl.Dense(input_shape[0]),
        # tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
    ])

    vae = tfk.Model(inputs=encoder.inputs,
                    outputs=decoder(encoder.outputs[0]))


    # negloglik = lambda x, rv_x: -rv_x.log_prob(x)

    vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                loss=tfk.losses.mse)

    return vae, encoder, decoder



# %%

if 0:
    latent_dim = 2

def sampling(args):
    
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

def vae_loss(inp, original, out, z_log_sigma, z_mean):
    
    reconstruction = K.mean(K.square(original - out)) * sequence_length
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

    return reconstruction + kl

def get_model():
    
    ### encoder ###
    
    inp = Input(shape=(sequence_length, 1))
    inp_original = Input(shape=(sequence_length, 1))
    
    cat_inp = []
    cat_emb = []
    for cat,i in map_col.items():
        inp_c = Input(shape=(sequence_length,))
        if cat in ['holiday', 'weather_main', 'weather_description']:
            emb = Embedding(X[cat].max()+2, 6)(inp_c)
        else:
            emb = Embedding(X[cat].max()+1, 6)(inp_c)
        cat_inp.append(inp_c)
        cat_emb.append(emb)
    
    concat = Concatenate()(cat_emb + [inp])
    enc = LSTM(64)(concat)
    
    z = Dense(32, activation="relu")(enc)
        
    z_mean = Dense(latent_dim)(z)
    z_log_sigma = Dense(latent_dim)(z)
            
    encoder = Model(cat_inp + [inp], [z_mean, z_log_sigma])
    
    ### decoder ###
    
    inp_z = Input(shape=(latent_dim,))

    dec = RepeatVector(sequence_length)(inp_z)
    dec = Concatenate()([dec] + cat_emb)
    dec = LSTM(64, return_sequences=True)(dec)
    
    out = TimeDistributed(Dense(1))(dec)
    
    decoder = Model([inp_z] + cat_inp, out)   
    
    ### encoder + decoder ###
    
    z_mean, z_log_sigma = encoder(cat_inp + [inp])
    z = Lambda(sampling)([z_mean, z_log_sigma])
    pred = decoder([z] + cat_inp)
    
    vae = Model(cat_inp + [inp, inp_original], pred)
    vae.add_loss(vae_loss(inp, inp_original, pred, z_log_sigma, z_mean))
    vae.compile(loss=None, optimizer=Adam(lr=1e-3))
    
    return vae, encoder, decoder