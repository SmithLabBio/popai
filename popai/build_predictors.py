"""Build predictive models."""
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.decomposition import PCA
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from memory_profiler import profile
from popai.dataset import PopaiTrainingData
from typing import Dict


class RandomForestsSFS:
    """Build a RF predictor that takes the SFS as input."""

    def __init__(self, config, simulations, subset, user=False):
        self.config = config
        self.arraydict, self.sfs, self.labels, self.label_to_int, self.int_to_label, self.nclasses = read_data(simulations, subset, user, type='1d')
        self.rng = np.random.default_rng(self.config['seed'])

    def build_rf_sfs(self, ntrees=500):
        """Build a random forest classifier that takes the
        multidimensional SFS as input."""

        train_test_seed = self.rng.integers(2**32, size=1)[0]
        x_train, x_test, y_train, y_test = train_test_split(self.sfs,
                self.labels, test_size=0.2, random_state=train_test_seed, stratify=self.labels)

        sfs_rf = RandomForestClassifier(n_estimators=ntrees, oob_score=True)
        sfs_rf.fit(x_train, y_train)
        print("Out-of-Bag (OOB) Error:", 1.0 - sfs_rf.oob_score_)

        # Convert predictions and true labels back to original labels
        y_test_pred = sfs_rf.predict(x_test)
        y_test_original = [self.int_to_label[label] for label in np.argmax(y_test, axis=1)]
        y_pred_original = [self.int_to_label[label] for label in np.argmax(y_test_pred, axis=1)]

        conf_matrix, conf_matrix_plot = plot_confusion_matrix(y_test_original, y_pred_original, labels=list(self.int_to_label.values()))

        return sfs_rf, conf_matrix, conf_matrix_plot

    def predict(self, model, new_data):
        new_data = np.array(new_data)
        predicted_prob = np.array(model.predict_proba(new_data))[:,:, 1].T

        if predicted_prob.shape[1] != self.nclasses:
            raise ValueError(f"Model has {predicted_prob.shape[1]} classes, but the provided data has {self.nclasses} classes. You probably used different subsets for training and applying.")
        
        headers = [f"Model {self.int_to_label[i]}" for i in range(len(model.classes_))]
        replicate_numbers = ["Replicate {}".format(i+1) for i in range(len(predicted_prob))]

        table_data = np.column_stack((replicate_numbers, predicted_prob))
        tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        return(tabulated)


class CnnSFS(keras.Model):
    """
    CNN taking two dimensional site frequency spectra for each population pair as input.
    """
    def __init__(self, n_pairs:int, n_classes:int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_pairs = n_pairs
        self.n_classes = n_classes
        self.conv1_layers = [] 
        for i in range(n_pairs):
            conv_layer = keras.layers.Conv2D(10, (3,3), activation="relu")
            self.conv1_layers.append(conv_layer)
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(64, activation="relu")
        self.dense2 = keras.layers.Dense(n_classes, activation="softmax")

    def call(self, x):
        outputs = []
        for i in range(self.n_pairs):
            out = self.conv1_layers[i](tf.expand_dims(x[:,i], axis=-1)) # Get each population pair, (batch, pair)
            out = self.flat(out)
            outputs.append(out)
        out = keras.layers.concatenate(outputs)
        out = self.dense1(out)
        out = self.dense2(out)
        return out
    
    # def get_feature_extractor(self):

    
    def get_config(self):
        # For saving model
        config = super().get_config()
        config.update(dict(
            n_pairs=self.n_pairs,
            n_classes=self.n_classes))
        return config

    @classmethod
    def from_config(cls, config):
        # For reading model from file
        n_pairs = config.pop("n_pairs")
        n_classes = config.pop("n_classes")
        return (cls(n_pairs, n_classes, **config))

class CnnNpy(keras.Model):
    """
    CNN taking SNP alignment as input.
    """
    def __init__(self, n_sites: int, sampling_dict: Dict[str,int], n_classes: int, name=None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_sites = n_sites
        self.sampling_dict = sampling_dict
        self.n_classes = n_classes
        self.conv1_layers = []
        self.rows = []
        for key, num_rows in sampling_dict.items():
            self.rows.append(num_rows)
            conv_layer = keras.layers.Conv2D(10, (num_rows, 1), strides=(num_rows, 1), 
                    activation="relu", padding="valid")
            self.conv1_layers.append(conv_layer)
        self.conv2 = keras.layers.Conv2D(10, (len(sampling_dict), 1), activation="relu", 
                                         padding="valid")
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.drop = keras.layers.Dropout(0.1)
        self.dense2 = keras.layers.Dense(50, activation='relu')
        self.dense3 = keras.layers.Dense(n_classes, activation="softmax")
    
    def call(self, x):
        x = tf.cast(tf.expand_dims(x, axis=-1), dtype=tf.float64) # Reshape input and cast to float dtype
        outputs = []
        start_ix = 0
        for i in range(len(self.rows)):
            num_rows = self.rows[i]
            end_ix = start_ix + num_rows 
            out = self.conv1_layers[i](x[:,start_ix:end_ix,:,:]) 
            outputs.append(out)
        out = keras.layers.concatenate(outputs, axis=1)
        out = self.conv2(out)
        out = self.flat(out)
        out = self.dense1(out)
        out = self.drop(out)
        out = self.dense2(out)
        out = self.dense3(out)
        return out

    def get_config(self):
        # For saving model
        config = super().get_config()
        config.update(dict(
            n_sites=self.n_sites, 
            downsampling_dict=self.downsampling_dict,
            n_classes=self.n_classes))
        return config

    @classmethod
    def from_config(cls, config):
        # For reading model from file
        n_sites = config.pop("n_sites")
        sampling_dict = config.pop("sampling_dict")
        n_classes = config.pop("n_classes")
        return (cls(n_sites, sampling_dict, n_classes, **config))


class NeuralNetSFS(keras.Model):
    """
    Fully connected neural network taking multidimensional site frequency spectrum as input.
    """
    def __init__(self, n_classes: int, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.fc1 = keras.layers.Dense(100, activation="relu")
        self.fc2 = keras.layers.Dense(50, activation="relu")
        self.fc3 = keras.layers.Dense(n_classes, activation="softmax")
    
    def call(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
    # def get_features_extractor(self):

    
    def get_config(self):
        # For saving model
        config = super().get_config()
        config.update(dict(n_classes=self.n_classes))
        return config

    @classmethod
    # For reading model from file
    def from_config(cls, config):
        n_classes = config.pop("n_classes")
        return (cls(n_classes, **config))


def train_model(model:keras.Model, data:PopaiTrainingData, outdir:str, label:str):
    """
    Run model training.
    model: Keras model
    data: PopaiTrainingData 
    outdir: Path to output directory
    label: label to prepend to output file
    """
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(data.train_loader, epochs=1, batch_size=10)#, validation_data=data.test_loader)
    model.save(os.path.join(outdir, f"{label}.keras"))
    print(model.summary())
    print(model.layers)
    print(model.get_layer("conv2d"))
    # model = keras.Model(model.get_layers[0], model.layers[-2])
    # extract the features
    # features = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    # features.save(os.path.join(outdir, f"{label}_featureextractor.keras"))

def test_model(model:keras.Model, data:PopaiTrainingData, outdir:str, label:str):
    """
    Run model training.
    model: Keras model
    data: PopaiTrainingData 
    outdir: Path to output directory
    label: label to prepend to output file
    """
    y_true = [data.dataset.labels[i] for i in data.test_dataset.indices]
    y_hat = model.predict(data.test_loader)
    y_pred = np.argmax(y_hat, axis=1).tolist()
    cm, cm_plot = plot_confusion_matrix(y_true, y_pred, labels=[str(i) for i in y_true]) 
    cm_plot.savefig(os.path.join(outdir, f"{label}_confusion.png"))

def predict(model_dir:keras.Model, model_file:str, data, out_dir:str, label:str):
    """
    Run model prediction with empirical data.
    model: Keras model
    model_file: Path to stored Keras model file
    data: PopaiTrainingData 
    outdir: Path to output directory
    label: label to prepend to output file
    """
    model = keras.models.load_model(os.path.join(model_dir, model_file))
    pred = model.predict(data)
    with open(os.path.join(out_dir, f"{label}_predictions.txt"), 'w') as fh:
        fh.write(str(pred))
    
    

def plot_confusion_matrix(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')#, xticklabels=labels, yticklabels=labels) # TODO: Fix this. 
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    return conf_matrix, plt

def check_valid_labels(labels):
    unique_labels = list(set(labels))
    unique_labels.sort()
    for i in range(len(unique_labels)):
        if unique_labels[i] !=i:
            return False
    return True

def read_data(simulations, subset, user, type):

    arraydict = {}
    sfs = []
    labels = []

    # read in the data
    if subset:
        subset_list = []
        with open(subset, 'r') as f:
            for line in f:
                subset_list.append(line.strip())
    pickle_list = os.listdir(simulations)
    if type=='1d':
        pickle_list = [x for x in pickle_list if ('_mSFS' in x and x.endswith('.pickle'))]
    elif type=='2d':
        pickle_list = [x for x in pickle_list if ('_2dSFS' in x and x.endswith('.pickle'))]
    elif type=='npy':
        pickle_list = [x for x in pickle_list if ('_arrays' in x and x.endswith('.pickle'))]
    if subset:
        pickle_list = [x for x in pickle_list if x.split('_')[-1].split('.')[0] in subset_list]
    pickle_list = sorted(pickle_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for item in pickle_list:
        modno = item.split('_')[-1].split('.')[0]
        if type=='1d':
            if user==True:
                with open(os.path.join(simulations, 'simulated_mSFS_model_%s.pickle' % str(modno)), 'rb') as f:
                    arraydict[str(modno)] = pickle.load(f)
            else:
                with open(os.path.join(simulations, 'simulated_mSFS_%s.pickle' % str(modno)), 'rb') as f:
                    arraydict[str(modno)] = pickle.load(f)
        elif type=='2d':
            if user==True:
                with open(os.path.join(simulations, 'simulated_2dSFS_model_%s.pickle' % str(modno)), 'rb') as f:
                    arraydict[str(modno)] = pickle.load(f)
            else:
                with open(os.path.join(simulations, 'simulated_mSFS_%s.pickle' % str(modno)), 'rb') as f:
                    arraydict[str(modno)] = pickle.load(f)
        elif type=='npy':
            if user==True:
                with open(os.path.join(simulations, 'simulated_arrays_model_%s.pickle' % str(modno)), 'rb') as f:
                    arraydict[str(modno)] = pickle.load(f)
            else:
                with open(os.path.join(simulations, 'simulated_mSFS_%s.pickle' % str(modno)), 'rb') as f:
                    arraydict[str(modno)] = pickle.load(f)

    for key,value in arraydict.items():
        for thearray in value:
            sfs.append(thearray)
            labels.append(key)
    nclasses = len(set(labels))
    if user:
        try:
            labels = [int(x.split('_')[-1]) for x in labels]
        except:
            raise ValueError(f"Model names must be 'Model_x', where x are integers ranging from 0 to n-1, where n is the number of models.")
    else:
        labels = [int(x) for x in labels]

    # Create a mapping from original labels to continuous integers
    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    
    # Convert labels to continuous integers
    labels = [label_to_int[label] for label in labels]
    
    # One-hot encode the labels
    labels = keras.utils.to_categorical(labels)

    # array
    sfs = np.array(sfs)

    return arraydict, sfs, labels, label_to_int, int_to_label, nclasses
