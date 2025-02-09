import re
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from utils import get_logger
import golois

logger = get_logger()


def choose_model(
    model_type,
    logger,
    regularizer_rate,
    planes,
    filters,
    kernel_size,
    nbr_layers,
):
    logger.info(f"Model type: {model_type}")
    if model_type == "unet":
        return create_unet_model(
            regularizer_rate=regularizer_rate,
            planes=planes,
            filters=filters,
            max_pooling_size=(2, 2),
            kernel_size=2,
        )
    elif model_type == "original":
        return create_original_model(
            planes=planes,
            filters=filters,
            kernel_size=kernel_size,
            regularizer_rate=regularizer_rate,
            nbr_layers=nbr_layers,
        )
    elif model_type == "v2":
        return create_model_2(
            planes,
            nbr_layers=8,
            filters=filters,
            kernel_size=kernel_size,
        )
    elif model_type == "resnet":
        return create_resnet(
            filters, planes, kernel_size, nbr_layers, regularizer_rate
        )
    else:
        raise ValueError("Model type not found")


def create_unet_model(
    planes,
    filters,
    regularizer_rate,
    kernel_size=(3, 3),
    max_pooling_size=(2, 2),
):
    inputs = keras.Input(shape=(19, 19, planes), name="board")

    # Encoder
    c1 = layers.Conv2D(
        filters, kernel_size, activation="relu", padding="same"
    )(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(
        filters, kernel_size, activation="relu", padding="same"
    )(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D(max_pooling_size)(c1)

    c2 = layers.Conv2D(
        filters * 2, kernel_size, activation="relu", padding="same"
    )(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(
        filters * 2, kernel_size, activation="relu", padding="same"
    )(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D(max_pooling_size)(c2)

    # c3 = layers.Conv2D(
    #     filters * 4, kernel_size, activation="relu", padding="same"
    # )(p2)
    # c3 = layers.BatchNormalization()(c3)
    # c3 = layers.Conv2D(
    #     filters * 4, kernel_size, activation="relu", padding="same"
    # )(c3)
    # c3 = layers.BatchNormalization()(c3)

    # Decoder
    u4 = layers.Conv2DTranspose(
        filters * 2, max_pooling_size, strides=max_pooling_size, padding="same"
    )(c2)
    u4 = layers.BatchNormalization()(u4)
    c1 = layers.Cropping2D(((1, 0), (1, 0)))(c1)
    u4 = layers.concatenate([u4, c1])
    c4 = layers.Conv2D(
        filters * 2, kernel_size, activation="relu", padding="same"
    )(u4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(
        filters * 2, kernel_size, activation="relu", padding="same"
    )(c4)
    c4 = layers.BatchNormalization()(c4)

    # u5 = layers.Conv2DTranspose(
    #     filters, max_pooling_size, strides=max_pooling_size, padding="same"
    # )(c4)
    # u5 = layers.BatchNormalization()(u5)
    # c1 = layers.Cropping2D(((3, 0), (3, 0)))(c1)
    # u5 = layers.concatenate([u5, c1])
    # c5 = layers.Conv2D(
    #     filters, kernel_size, activation="relu", padding="same"
    # )(u5)
    # c5 = layers.BatchNormalization()(c5)
    # c5 = layers.Conv2D(
    #     filters, kernel_size, activation="relu", padding="same"
    # )(c5)
    outputs = layers.BatchNormalization()(c4)

    # outputs = layers.Dense(
    #     361, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    # )(c4)

    policy_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.0001),
    )(outputs)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(
        361, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    )(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)

    value_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.0001),
    )(outputs)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    )(value_head)
    value_head = layers.Dense(
        1,
        activation="sigmoid",
        name="value",
        kernel_regularizer=regularizers.l2(0.0001),
    )(value_head)

    model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model


def create_model_2(planes, nbr_layers, filters=32, kernel_size=4):
    input = keras.Input(shape=(19, 19, planes), name="board")

    x = input
    mem = input

    for _ in range(nbr_layers):
        x = layers.Conv2D(
            filters, kernel_size, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(
            input.shape[-1], kernel_size, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, mem])

        x = layers.Conv2D(
            filters, kernel_size, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(
            input.shape[-1], kernel_size, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)

        mem = x

    policy_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.0001),
    )(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)
    value_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.00001),
    )(x)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(
        64, activation="relu", kernel_regularizer=regularizers.l2(0.00001)
    )(value_head)
    value_head = layers.Dense(
        1,
        activation="sigmoid",
        name="value",
        kernel_regularizer=regularizers.l2(0.0001),
    )(value_head)

    model = keras.Model(inputs=input, outputs=[policy_head, value_head])

    return model


def create_original_model(
    planes,
    filters,
    nbr_layers,
    regularizer_rate,
    kernel_size,
):
    input = keras.Input(shape=(19, 19, planes), name="board")

    x = layers.Conv2D(filters, 1, activation="relu", padding="same")(input)

    for _ in range(nbr_layers):
        x = layers.Conv2D(
            filters, kernel_size, activation="relu", padding="same"
        )(x)

    policy_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.0001),
    )(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)
    value_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.0001),
    )(x)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(0.0001)
    )(value_head)
    value_head = layers.Dense(
        1,
        activation="sigmoid",
        name="value",
        kernel_regularizer=regularizers.l2(0.0001),
    )(value_head)

    model = keras.Model(inputs=input, outputs=[policy_head, value_head])

    return model


def start_part_model(N, planes, moves):
    ## input format
    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    # normalize
    input_data = input_data.astype("float32")

    ## policy format
    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical(policy)

    value = np.random.randint(2, size=(N,))
    value = value.astype("float32")

    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype("float32")

    groups = np.zeros((N, 19, 19, 1))
    groups = groups.astype("float32")

    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    return input_data, policy, value, end, groups


def end_part_model(x, regularizer_rate):
    policy_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(x)

    policy_head = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(policy_head)

    policy_head = layers.Dense(
        1,
        activation="sigmoid",
        name="policy1",
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(policy_head)

    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)

    value_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(x)

    value_head = layers.Flatten()(value_head)

    value_head = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(value_head)

    value_head = layers.Dense(
        1,
        activation="sigmoid",
        name="value",
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(value_head)

    return policy_head, value_head


def create_resnet(filters, planes, kernel_size, nbr_layers, regularizer_rate):
    inputs = keras.Input(shape=(19, 19, planes), name="board")

    x = layers.BatchNormalization()(inputs)

    # Input Layer
    x = residual_input_layer(x, filters)

    # Residual Layers
    for _ in range(nbr_layers):
        x = residual_layer(x, filters, kernel_size)

    # Output Layer
    x = residual_output_layer(x, filters)

    # Policy layers
    policy_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)

    # Value layers
    value_head = layers.Conv2D(
        1,
        1,
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(x)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(
        50,
        activation="relu",
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(value_head)
    value_head = layers.Dense(
        1,
        activation="sigmoid",
        name="value",
        kernel_regularizer=regularizers.l2(regularizer_rate),
    )(value_head)

    model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model


def residual_layer(x, filters, kernel_size):
    # store the result
    mem = x

    # Normalization
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # RELU
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # addition
    x = layers.add([x, mem])

    # RELU
    x = layers.Activation("relu")(x)

    return x


def residual_input_layer(x, filters):
    x_left = layers.Conv2D(filters, 5, padding="same")(x)

    x_right = layers.Conv2D(filters, 1, padding="same")(x)

    # addition
    x = layers.add([x_left, x_right])

    # Activation
    x = layers.Activation("relu")(x)

    return x


def residual_output_layer(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)

    x = layers.Activation("softmax")(x)

    return x