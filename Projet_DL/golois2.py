import gc

from matplotlib.pyplot import hist
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, callbacks

import golois
import sys
import os

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("WebAgg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#import configuration

from utils import (
    append_dict_from_list,
    get_logger,
    unsqueeze_list,
    check_format,
    append_dict,
)

from model import (
    start_part_model,
    choose_model,
)

from display import plot_loss, plot_metrics

logger = get_logger()

# Configuration
planes = 31
moves = 361  # 361  # 351
N = 10000
epochs = 120
batch = 128

# Settings
filters = 31


# Optimization
learning_rate = 0.001
regularizer_rate = 0.0005
momentum = 0.5

# Original model
nbr_layers = 2
kernel_size = 3

# Architecture Golois

input_data, policy, value, end, groups = start_part_model(N, planes, moves)

# Architecture
model = choose_model(
    configuration.model_type,
    logger,
    regularizer_rate,
    planes,
    filters,
    kernel_size,
    nbr_layers,
)

model.summary()

# Check the number of parameters
logger.info(f"Number of parameters: {model.count_params()}")

# Stop the program if the number of parameters more than 100k
if model.count_params() > 100000:
    logger.error("Number of parameters is more than 100k")
    raise ValueError("Number of parameters is more than 100k")


# Utiliser l'optimiseur Adam
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Plan de réduction du taux d'apprentissage
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=5, min_lr=1e-6
)

model.compile(
    optimizer=optimizer,
    loss={
        "policy": "categorical_crossentropy",
        "value": "binary_crossentropy",
    },
    loss_weights={"policy": 1.0, "value": 1.0},
    metrics={
        "policy": ["categorical_accuracy", "accuracy"],
        "value": ["mse", "accuracy"],
    },
)

history = None
history_train = {
    "loss": [],
    "policy_loss": [],
    "value_loss": [],
    "policy_categorical_accuracy": [],
    "value_mse": [],
    "policy_accuracy": [],
}

history_val = history_train.copy()

for i in range(1, epochs + 1):
    print("epoch " + str(i))
    golois.getBatch(input_data, policy, value, end, groups, i * N)
    history = model.fit(
        input_data,
        {"policy": policy, "value": value},
        epochs=1,
        batch_size=batch,
    )

    if i % 5 == 0:
        gc.collect()
    if i % 3 == 0:
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(
            input_data, [policy, value], verbose=0, batch_size=batch
        )

        append_dict(history_train, history.history)

        append_dict_from_list(history_val, val)

        print("val =", val)
        model.save("checkpoint_" + configuration.name_checkpoint + ".h5")

check_format(history_train)
check_format(history_val)

plot_loss(history_train, history_val)
plot_metrics(history_train)


plt.show()