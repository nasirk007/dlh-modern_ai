#!/usr/bin/env python3
"""
train image classifier with transfer learning on Caltech-101
"""
import tensorflow as tf
from tensorflow import keras


def train_transfer_model():
    """
    Train, fine-tune, and save a
    Caltech-101 image classifier
    """

    d_dir = '101_ObjectCategories'
    vs = 0.2
    sd = 42
    i_ze = (224, 224)
    b_ze = 32

    train_data = keras.utils.image_dataset_from_directory(
        data_dir=d_dir,
        validation_split=vs,
        subset='training',
        seed=sd,
        image_size=i_ze,
        batch_size=b_ze
    )

    vld_data = keras.utils.image_dataset_from_directory(
        data_dir=d_dir,
        validation_split=vs,
        subset='validation',
        seed=sd,
        image_size=i_ze,
        batch_size=b_ze
    )

    num_classes = len(train_data.class_names)

    augmtn = keras.Sequential()
    augmtn.add(keras.layers.RandomFlip(
        'horizontal', seed=sd
    ))
    augmtn.add(keras.layers.RandomRotation(0.15, seed=sd))
    augmtn.add(keras.layers.RandomZoom(0.15, seed=sd))
    augmtn.add(keras.layers.RandomContrast(0.1, seed=sd))

    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = augmtn(inputs)
    x = keras.activations.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    model = keras.Model(inputs, outputs)

    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    validation_data = vld_data.prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=15,
        callbacks=callbacks
    )

    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    for layer in base_model.layers[-20:]:
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        ):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=10,
        callbacks=callbacks
    )

    model.save('caltech101_model.h5')
