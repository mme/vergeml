from vergeml import ModelPlugin, model, train, predict, option, VergeMLError
from vergeml.display import DISPLAY
import numpy as np
import os
import os.path
import random

@model('imagenet', descr='Image classifier model, with weights pre-trained on ImageNet.')
class ImageNetModelPlugin(ModelPlugin):

    @train('train', descr='Train an image classifier.')
    @option('epochs', 5)
    @option('cnn', 'resnet-50', 'Name of the pretrained network.')
    @option('variant', '*auto*', 'Network variant.')
    @option('size', "*auto*", 'Image input size.')
    @option('alpha', 1.0, 'Network alpha value.')
    @option('layers', 1, 'Number of layers to add.')
    @option('output-layer', '*last*', 'Name or index of the output layer.', type='Union[int,str]')
    @option('batch-size', 64)
    @option('optimizer', 'sgd', validate=('adam', 'sgd'))
    @option('learning-rate', 0.0001)
    @option('decay', 0.)
    @option('dropout', 0.)
    @option('early-stopping-delta', 0.0, 'Early stopping delta.')
    @option('early-stopping-patience', 0, 'Early stopping patience (0 means off).')
    @option('name', None, type='Optional[str]', descr='Optional name of the AI.')

    # supported features:
    # - retrain a model
    #    . model name
    # - finetune
    #    . model name

    def train(self, args, env):

        from vergeml.sources.features import get_image_size, evaluate_args

        evaluate_args(args['cnn'], env.get('trainings-dir'), args['variant'], args['alpha'], args['size'])

        # configure libraries
        env.configure('keras')

        self.model = ImageNetModel()
        size = get_image_size(args['cnn'], args['variant'], args['size'])


        # gather arguments
        trainargs = dict(xy_train=env.data.load('train', view='batch', layout='arrays', batch_size=('dynamic', args['batch-size']),
                                                randomize=True, infinite=True),
                         xy_val=list(env.data.load('val', view='list', layout='arrays')),
                         xy_test=list(env.data.load('test', view='list', layout='arrays')),
                         labels=env.data.meta['labels'])
        # start training
        env.start_training(name=args['name'],
                           hyperparameters={'labels': trainargs['labels'],
                                            'size': size,
                                            'layers': args['layers'],
                                            'cnn': args['cnn']})

        trainargs.update(env.args_for(self.model.train, args))
        trainargs['callbacks'] = [env.keras_callback()]

        # train
        final_results = self.model.train(**trainargs)

        # done
        env.end_training(final_results)


    @predict('predict', descr="Predict image labels.")
    @option('@AI', type='AI')
    @option('labels', default=5, type=int, validate='>0', descr="The number of labels to predict.")
    @option('compact', default=False, descr="Show results in a compact representation.", flag=True, command_line=True)
    @option('<files>', type='List[file]', descr="The images to predict.")
    def predict(self, args, env):

        if not self.model:
            self.load(env)

        files = args['<files>']
        res = []

        for ix, f in enumerate(files):
            results = self.model.predict(f, k=args['labels'])
            res.append(results)
            if args['compact']:
                DISPLAY.print("{}\t{}".format(f, results['prediction'][0]['label']))
            else:
                DISPLAY.print(f)
                for pred in results['prediction']:
                    DISPLAY.print("{:>6.2f}%    {}".format(pred['probability']*100, pred['label']))
                if ix + 1 < len(files):
                    DISPLAY.print("")
        return res

    def load(self, env):
        env.configure('keras')
        # load the AI
        self.model = ImageNetModel()
        input_size = env.get("hyperparameters.size")
        cnn = env.get("hyperparameters.cnn")
        self.model.load(os.path.join(env.checkpoints_dir()), cnn, input_size)

    def set_defaults(self, cmd, args, env):
        if cmd == 'train':
            for k in ('input', 'output'):
                # the user can specify a different input, but the output has to be labeled-image-features
                type_ = env.get(f"data.{k}.type")
                if (k == 'input' and type_ in (None, 'labeled-image-features')) or k == 'output':
                    env.set(f"data.{k}.type", 'labeled-image-features')
                    env.set(f"data.{k}.output-layer", args['output-layer'])
                    env.set(f"data.{k}.cnn", args['cnn'])
                    env.set(f"data.{k}.variant", args['variant'])
                    env.set(f"data.{k}.size", args['size'])
                    env.set(f"data.{k}.alpha", args['alpha'])


class ImageNetModel:

    labels = None
    model = None
    trained_model = None
    base_model = None
    image_size = None
    preprocess_input = None

    def load(self, model_dir, cnn, image_size):
        from keras.models import load_model
        from vergeml.sources.features import get_preprocess_input
        labels_txt = os.path.join(model_dir, "labels.txt")
        if not os.path.exists(labels_txt):
            raise VergeMLError("labels.txt not found: {}".format(labels_txt))

        model_h5 = os.path.join(model_dir, "model.h5")
        if not os.path.exists(model_h5):
            raise VergeMLError("model.h5 not found: {}".format(model_h5))

        with open(labels_txt, "r") as f:
            self.labels = f.read().splitlines()

        self.model = load_model(model_h5)
        self.image_size = image_size
        self.preprocess_input = get_preprocess_input(cnn)

    def train(self,
              labels,
              xy_train,
              xy_val,
              xy_test,
              epochs=20,
              batch_size=64,
              cnn="resnet-50",
              variant="*auto*",
              size="*auto*",
              alpha=1.0,
              layers=1,
              output_layer="*last*",
              optimizer='sgd',
              learning_rate=0.0001,
              decay=0.,
              dropout=0.,
              early_stopping_delta=0.,
              early_stopping_patience=0,
              random_seed=2204,
              callbacks=[],
              trainings_dir="trainings",
              checkpoints_dir="checkpoints",
              stats_dir='stats'):

        from vergeml.sources.features import get_imagenet_cnn, get_image_size, get_preprocess_input
        from keras.layers import Dense, Input
        from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
        from keras.models import Model
        from keras import optimizers

        if cnn.startswith("@"):
            raise NotImplementedError

        self.labels = labels
        nclasses = len(self.labels)
        num_batches = len(xy_train)
        self.image_size = get_image_size(cnn, variant, size)
        self.preprocess_input = get_preprocess_input(cnn)

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        self.base_model = get_imagenet_cnn(cnn, variant, size, alpha, output_layer, include_top=False, weights='imagenet')
        input_size = np.array(self.base_model.layers[-1].output_shape[1:]).prod()
        input_layer = Input(shape=(input_size,))

        x = _makenet(input_layer, layers, dropout, random_seed)

        output_layer = Dense(nclasses, activation="softmax", name="predictions")(x)
        self.trained_model = Model(input=input_layer, output=output_layer)

        if optimizer == 'adam':
            self.trained_model.compile(loss='categorical_crossentropy',
                                       optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
                                       metrics=['accuracy'])
        else:
            self.trained_model.compile(loss="categorical_crossentropy",
                                       optimizer=optimizers.SGD(lr=learning_rate, decay=decay, momentum=0.9),
                                       metrics=["accuracy"])

        callbacks = callbacks.copy()

        has_val_step = bool(len(xy_val[0]))

        # during training, save the new layers only
        checkpoint = ModelCheckpoint(os.path.join(checkpoints_dir, "last_layers.h5"),
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=has_val_step,
                                     save_weights_only=True,
                                     mode='auto',
                                     period=1)
        callbacks.append(checkpoint)

        if early_stopping_delta:
            callbacks.append(EarlyStopping(min_delta=early_stopping_delta, patience=early_stopping_patience))

        callbacks.append(TensorBoard(log_dir=stats_dir))



        try:
            self.trained_model.fit_generator(xy_train,
                                             epochs=epochs,
                                             verbose=0,
                                             validation_data=xy_val if has_val_step else None,
                                             steps_per_epoch=num_batches,
                                             callbacks=callbacks)

        except KeyboardInterrupt:
            pass

        history = self.trained_model.history  # pylint: disable=E1101

        final_results = {}
        if hasattr(history, 'epoch') and len(history.epoch):
            # load the best weights
            self.trained_model.load_weights(os.path.join(checkpoints_dir, "last_layers.h5"))

            final_results = _evaluate_final(self.trained_model, xy_test, batch_size, history)

            self.model = _save(self.trained_model, self.base_model, layers, labels, random_seed, checkpoints_dir)
        return final_results

    def predict(self, f, k=5):
        from keras.preprocessing import image

        filename = os.path.basename(f)

        if not os.path.exists(f):
            return dict(filename=filename, prediction=[])

        # TODO don't distort
        img = image.load_img(f, target_size=(self.image_size, self.image_size))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        preds = self.model.predict(x)
        pred = self._decode(preds, top=k)[0]
        prediction=[dict(probability=np.asscalar(perc), label=klass) for _, klass, perc in pred]

        return dict(filename=filename, prediction=prediction)

    def _decode(self, preds, top):
        preds = list(preds[0])
        dec = list(zip([None] * len(self.labels), self.labels, preds))
        dec = sorted(dec, key=lambda x: x[2], reverse=True)
        return [dec[:top]]


def _evaluate_final(model, xy_test, batch_size, history):
    res = {}

    if 'val_acc' in history.history:
        res['val_acc'] = max(history.history['val_acc'])
        rev_ix = -1 - list(reversed(history.history['val_acc'])).index(res['val_acc'])
        res['val_loss'] = history.history['val_loss'][rev_ix]

    res['acc'] = history.history['acc'][-1]
    res['loss'] = history.history['loss'][-1]

    if len(xy_test[0]):
        # evaluate with test data
        x_test, y_test = xy_test
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        res['test_loss'] = test_loss
        res['test_acc'] = test_acc

    return res


def _makenet(x, num_layers, dropout, random_seed):
    from keras.layers import Dense, Dropout

    dropout_seeder = random.Random(random_seed)

    for i in range(num_layers - 1):
        # add intermediate layers
        if dropout:
            x = Dropout(dropout, seed=dropout_seeder.randint(0, 10000))(x)
        x = Dense(1024, activation="relu", name='dense_layer_{}'.format(i))(x)

    if dropout:
        # add the final dropout layer
        x = Dropout(dropout, seed=dropout_seeder.randint(0, 10000))(x)

    return x

def _save(model, base_model, layers, labels, random_seed, checkpoints_dir):
    from keras.layers import Flatten, Dense
    from keras import Model
    nclasses = len(labels)
    x = Flatten()(base_model.output)
    x = _makenet(x, layers, dropout=None, random_seed=random_seed)
    predictions = Dense(nclasses, activation="softmax", name="predictions")(x)
    model_final = Model(inputs=base_model.input, outputs=predictions)

    for i in range(layers - 1):
        weights = model.get_layer(name='dense_layer_{}'.format(i)).get_weights()
        model_final.get_layer(name='dense_layer_{}'.format(i)).set_weights(weights)

    weights = model.get_layer(name='predictions').get_weights()
    model_final.get_layer(name='predictions').set_weights(weights)

    model_final.save(os.path.join(checkpoints_dir, "model.h5"))
    with open(os.path.join(checkpoints_dir, "labels.txt"), "w") as f:
        f.write("\n".join(labels))
    return model_final