#!/usr/bin/env python3
"""
Build a data augmentation pipeline
"""
import tensorflow as tf


def build_data_augmentation():
    """
    Return a seeded Sequential image augumentation model
    """

    seed = 42
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            'horizontal',
            seed=seed
        ),
        tf.keras.layers.RandomRotation(
            0.15,
            seed=seed
        ),
        tf.keras.layers.RandomZoom(
            0.15,
            seed=seed
        ),
        tf.keras.layers.RandomContrast(
            0.1,
            seed=seed
        )
    ])
