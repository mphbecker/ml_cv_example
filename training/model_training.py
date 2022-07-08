import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
import os
import sys
import yaml

p = os.path.abspath('.')
sys.path.insert(1, p)
from models.lenet5 import LeNet
from data_import_and_preprocessing.dataset_formation import DataParser, ImageDataExtractor, LabelExtractor, \
    DataSetCreator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    with open('params.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    image_height = params['data']['image_height']
    image_weight = params['data']['image_width']
    color_channels = 1
    no_classes = params['data']['no_classes']
    no_epochs = params['training']['no_epochs']
    batch_size = params['training']['batch_size']
    loss = params['training']['loss']
    lr = params['training']['learning_rate']

    data_dir = 'data/preprocessed'
    data_parser = DataParser(data_dir)
    image_data_extractor = ImageDataExtractor((image_height, image_weight, color_channels))
    label_extractor = LabelExtractor(no_classes=no_classes)
    dataset = DataSetCreator(data_parser, image_data_extractor, label_extractor, no_repeats=no_epochs)
    no_points = len(dataset)
    no_points_train = int(no_points * 0.8)
    no_points_val = int(no_points * 0.1)

    dataset_train = dataset.take(no_points=no_points_train)
    dataset_val = dataset.take(starting_point=no_points_train + 1, no_points=no_points_val)
    dataset_test = dataset.take(starting_point=no_points_train + no_points_val + 2,
                                no_points=no_points - no_points_train - no_points_val)
    dataset_test.no_repeats = 1

    addit_info = {'data_shape': dataset_train.data_signature_output,
                  'label_shape': dataset_train.label_signature_output,
                  'no_points': no_points,
                  'no_points_train': no_points_train,
                  'no_points_val': no_points_val,
                  'no_points_test': no_points - no_points_train - no_points_val,
                  'data_points_train': [datapoint.datapoint_id for datapoint in dataset_train.data_points],
                  'data_points_val': [datapoint.datapoint_id for datapoint in dataset_val.data_points],
                  'data_points_test': [datapoint.datapoint_id for datapoint in dataset_test.data_points]}

    with open('cache/additional_info.yaml', 'w') as outfile:
        yaml.dump(addit_info, outfile, default_flow_style=False, sort_keys=False)

    tf_dataset_train = dataset_train.cast_tf_dataset().batch(batch_size).prefetch(1)
    tf_dataset_val = dataset_val.cast_tf_dataset().batch(batch_size).prefetch(1)
    tf_dataset_test = dataset_test.cast_tf_dataset().batch(batch_size)

    if not os.path.exists('cache'):
        os.makedirs('cache')
    tf.data.experimental.save(tf_dataset_test, 'cache/ds_test.tf')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(image_height, image_weight, color_channels)
    )

    base_model.trainable = False
    inputs = keras.Input(shape=(image_height, image_weight, color_channels))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    
    #model_built = model.build(data_shape=dataset_train.data_signature_output,
    #
    #                           label_shape=dataset_train.label_signature_output)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      patience=3,
                                                      restore_best_weights=True)

    log_folder = 'training/results/train_logs'
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder,
                                                 histogram_freq=1,
                                                 write_graph=True,
                                                 write_images=True,
                                                 update_freq='epoch',
                                                 profile_batch=2,
                                                 embeddings_freq=1)

    # Callback to save model after each batch
    checkpoint_path = 'training/results/checkpoints/checkpoint-{epoch:04d}.ckpt'
    save_every_epoch = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_freq=no_points_train // batch_size * 25)
    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit(tf_dataset_train,
                              epochs=no_epochs,
                              batch_size=batch_size,
                              steps_per_epoch=no_points_train // batch_size,
                              validation_data=tf_dataset_val,
                              validation_steps=no_points_val // batch_size ,
                              callbacks=[tensorboard, save_every_epoch],
                              )

    model.save('training/model/trained_model.h5')

    #with open('training/results/training_report.json', 'w', encoding='utf-8') as f:
    #    json.dump(history.history, f, ensure_ascii=False, indent=4)
