"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.
Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
pip install git+git://github.com/keras-team/keras.git --upgrade --no-deps
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
pip install tensorflow-gpu --upgrade

sudo nvidia-docker run -it -p 4444:8889 -p 3003:6006 -v /sharedfolder:/root/sharedfoler jihong/keras-gpu  bash
"""
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D,BatchNormalization,Activation,Input,MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from data import DataSet
import os.path

data = DataSet()

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'inception.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)
	
    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
    return train_generator, validation_generator
# Reference - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x
def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    # B1
    x = conv2d_bn(x, 96, 7, 7, padding='same')
    x = MaxPooling2D((2,2), strides=(2, 2))(x)
    #  B2 
    x = conv2d_bn(x, 256, 5, 5, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # B3 
 
    x = conv2d_bn(x,512 , 3, 3, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    '''
    # B5
    x = conv2d_bn(x, 256, 5, 5, padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    '''
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(4096, activation='relu')(x)
	x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    
    # and a logistic layer
    predictions = Dense(len(data.classes), activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model
def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
		        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model
def main(weights_file):
    model = get_model()
    generators = get_generators()
    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = freeze_all_but_top(model)
        model = train_model(model, 10, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)
    # Get and train the mid layers.
    model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 5, generators,
                        [checkpointer, early_stopper, tensorboard])
if __name__ == '__main__':
    weights_file = None
main(weights_file)
