import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, MaxPool2D, GlobalAveragePooling2D, Reshape, Permute, Lambda, Flatten, Activation
import gc

import golois

planes = 31
moves = 361
N = 10000
epochs = 30
batch = 128


input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)

value = np.random.randint(2, size=(N,))
value = value.astype ('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')





filters = 84
trunk = 32

blocks = 15


def lr (min,max,nb_steps):
    eta_min = 0
    eta_max = 0
    T_cur = 0
    T_i = 0
    pi = np.pi
    learning_rate = 0.0001

    learning_rate = eta_min + (1/2)*(eta_max - eta_min)*(1+np.cos((T_cur/T_i)*pi))

    return learning_rate


def bottleneck_block(x, expand=filters, squeeze=trunk, learning_rate):
    m = layers.Conv2D(expand, (1,1),
    kernel_regularizer=regularizers.l2(0.0001),
    use_bias = False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)
    m = layers.DepthwiseConv2D((3,3), padding="same",
    use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)
    m = layers.Conv2D(squeeze, (1,1),
    kernel_regularizer=regularizers.l2(0.0001),
    use_bias = False)(m)
    m = layers.BatchNormalization()(m)
    return layers.Add()([m, x])




def getModel ():
    input = keras.Input(shape=(19, 19, 31), name="board")
    x = layers.Conv2D(trunk, 1, padding="same",
    kernel_regularizer=regularizers.l2(0.0001))(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for i in range (blocks):  # à modifier
        x = bottleneck_block (x, filters, trunk)
    policy_head = layers.Conv2D(1, 1, activation="relu", padding="same",
    use_bias = False,
    kernel_regularizer=regularizers.l2(0.0001))(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)
    value_head = layers.GlobalAveragePooling2D()(x)
    value_head = layers.Dense(50, activation="relu",
    kernel_regularizer=regularizers.l2(0.0001))(value_head)
    value_head = layers.Dense(1, activation="sigmoid", name="value",
    kernel_regularizer=regularizers.l2(0.0001))(value_head)
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model

print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)

model_go = getModel()
print("le model summary : \n")
model_go.summary()

model_go.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups, i * N)
    history = model_go.fit(input_data,
                        {'policy': policy, 'value': value},
                        epochs=1, batch_size=batch)
    if (i % 5 == 0):
        gc.collect ()
    if (i % 20 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model_go.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)
       # model_go.save ('Brice_Charles.h5')
