"""Build predictive models."""
from sklearn.model_selection import train_test_split
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
from popai.dataset import PopaiTrainingData, human_sort_key
from typing import Dict
import glob
from torch.utils.data import DataLoader
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable

class RandomForestsSFS:
    """Build a RF predictor that takes the SFS as input."""

    def __init__(self, config, simulations):
        self.config = config
        self.arraydict, self.sfs, self.labels, self.label_to_int, self.int_to_label, self.nclasses = read_data(simulations)
        self.rng = np.random.default_rng(self.config['seed'])

    def build_rf_sfs(self, ntrees=500):
        """Build a random forest classifier that takes the
        multidimensional SFS as input."""

        train_test_seed = self.rng.integers(2**32, size=1)[0]
        x_train, x_test, y_train, y_test = train_test_split(self.sfs,
                self.labels, test_size=0.2, random_state=train_test_seed, stratify=self.labels)

        sfs_rf = RandomForestClassifier(n_estimators=ntrees, oob_score=True,
                                          random_state=int(self.rng.integers(2**32)))
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

@register_keras_serializable(package="Custom")
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

    def call(self, x, return_intermediate=False):
        outputs = []
        for i in range(self.n_pairs):
            out = self.conv1_layers[i](tf.expand_dims(x[i], axis=-1)) # Get each population pair, (batch, pair)
            out = self.flat(out)
            outputs.append(out)
        out = keras.layers.concatenate(outputs)
        out = self.dense1(out)
        if return_intermediate:
            return out  # Return feature map before further processing
        out = self.dense2(out)
        return out



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

@register_keras_serializable(package="Custom")
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
        self.conv1 = keras.layers.Conv2D(10, (3,3), activation="relu", 
                                         padding="valid")
        self.conv2 = keras.layers.Conv2D(10, (3,3), activation="relu", 
                                         padding="valid")        
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.drop = keras.layers.Dropout(0.1)
        self.dense2 = keras.layers.Dense(50, activation='relu')
        self.dense3 = keras.layers.Dense(n_classes, activation="softmax")

        self.pooling = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
        self.globalpool = keras.layers.GlobalAveragePooling2D()

    def call(self, x, return_intermediate=False):
        x = tf.cast(tf.expand_dims(x, axis=-1), dtype=tf.float64) # Reshape input and cast to float dtype
        out = self.conv1(x)
        out = self.pooling(out)
        out = self.conv2(out)
        out = self.pooling(out)

        out = self.flat(out)
        out = self.dense1(out)
        out = self.drop(out)
        out = self.dense2(out)
        if return_intermediate:
            return out  # Return feature map before further processing
 
        out = self.dense3(out)
        return out

    def get_config(self):
        # For saving model
        config = super().get_config()
        config.update(dict(
            n_sites=self.n_sites, 
            sampling_dict=self.sampling_dict,
            n_classes=self.n_classes))
        return config

    @classmethod
    def from_config(cls, config):
        # For reading model from file
        n_sites = config.pop("n_sites")
        sampling_dict = config.pop("sampling_dict")
        n_classes = config.pop("n_classes")
        return (cls(n_sites, sampling_dict, n_classes, **config))

@register_keras_serializable(package="Custom")
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
    
    def call(self, x, return_intermediate=False):
        out = self.fc1(x)
        out = self.fc2(out)
        if return_intermediate:
            return out  # Return feature map before further processing
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

def train_model(model:keras.Model, data:PopaiTrainingData, outdir:str, label:str, epochs:int=10, batch_size:int=10, learning_rate:float=0.001):
    """
    Run model training.
    model: Keras model
    data: PopaiTrainingData 
    outdir: Path to output directory
    label: label to prepend to output file
    """

    optimizer = Adam(learning_rate = learning_rate)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(data.train_loader, epochs=epochs, validation_data=data.test_loader, shuffle=False)
    model.save(os.path.join(outdir, f"{label}.keras"))

    # extract and save features
    extracted_features, all_labels = extract_features(model, data.train_loader, label)
    features_path = os.path.join(outdir, f"{label}_features.npy")
    labels_path = os.path.join(outdir, f"{label}_labels.npy")
    np.save(features_path, extracted_features)
    np.save(labels_path, all_labels)

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
    names = [str(number) for number in data.dataset.model_numbers]   # model numbers, from the file names
    cm, cm_plot = plot_confusion_matrix(y_true, y_pred, labels=list(range(data.dataset.n_classes)), names=names)
    cm_plot.savefig(os.path.join(outdir, f"{label}_confusion.png"))
    cm_plot.close()
    write_confusion_tsv(cm, names, os.path.join(outdir, f"{label}_confusion.tsv"))

def extract_features(model, dataset, label):
    """Features from the layer before the output layer of a trained model. dataset is a
    DataLoader (returns the features and the one-hot labels, batch by batch) or one batch of
    inputs (returns the features)."""
    if isinstance(dataset, DataLoader):
        extracted_features, all_labels = [], []
        for x_batch, y_batch in dataset:
            extracted_features.append(model(x_batch, return_intermediate=True).numpy())
            all_labels.append(y_batch)
        return np.concatenate(extracted_features, axis=0), np.concatenate(all_labels, axis=0)
    return model(dataset, return_intermediate=True).numpy()

def predict(model_dir:str, model_file:str, data, out_dir:str, label:str, sim_dir:str, path:str, train_features:str, train_labels:str):
    """
    Run model prediction with empirical data.
    model: directory with model
    model_file: Path to stored Keras model file
    data: PopaiTrainingData 
    outdir: Path to output directory
    label: label to prepend to output file
    train_features: File name for training features
    train_labels: File name for training labels
    """

    train_features_path = os.path.join(model_dir, train_features)
    train_labels_path = os.path.join(model_dir, train_labels)


    model = keras.models.load_model(os.path.join(model_dir, model_file))

    if label == "cnn":
        # one array per population pair, (replicates, rows, columns), each SFS normalized to
        # sum to 1, as custom_collate_fn in dataset.py does for the simulated 2D SFS
        data = [np.stack([replicate[i] / np.sum(replicate[i]) for replicate in data]).astype(np.float32)
                for i in range(len(data[0]))]

    pred = model.predict(data)
    replicate_numbers = ["Replicate {}".format(i+1) for i in range(pred.shape[0])]

    labels = get_labels(sim_dir, path)
    headers = [f"Model {i}" for i in labels]
    table_data = np.column_stack((replicate_numbers, pred))
    tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

    with open(os.path.join(out_dir, f"{label}_predictions.txt"), 'w') as fh:
        fh.write(tabulated)

    new_features = extract_features(model, data, label)

    new_features_path = os.path.join(out_dir, f"{label}_empirical_features.npy")
    np.save(new_features_path, new_features)

    # load training
    train_features = np.load(train_features_path)
    train_labels = np.load(train_labels_path)

    # pca
    outputpca = os.path.join(out_dir, f"{label}_pca.png")
    plot_pca(train_features, np.array(labels)[np.argmax(train_labels, axis=1)], new_features, outputpca)

def plot_confusion_matrix(y_true, y_pred, labels, names=None):
    """Confusion matrix of y_true (rows) against y_pred (columns), with one row and column for
    each entry of labels, in that order. names label the rows and columns (default: labels)."""
    names = labels if names is None else names
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    return conf_matrix, plt

def write_confusion_tsv(conf_matrix, names, path):
    """Write a confusion matrix as a TSV: rows are the true models, columns the predicted models."""
    with open(path, "w") as f:
        f.write("\t" + "\t".join(names) + "\n")
        for name, row in zip(names, conf_matrix):
            f.write(name + "\t" + "\t".join(map(str, row)) + "\n")

def read_data(simulations):
    """The simulated mSFS, one row per replicate, and their one-hot labels. Reads every
    simulated_mSFS_<number>.pickle in simulations (named <number> or model_<number>); the
    numbers become classes 0 .. n-1 in numerical order, as in dataset.py."""
    paths = glob.glob(os.path.join(simulations, "simulated_mSFS_*.pickle"))
    if not paths:
        raise FileNotFoundError(f"No simulated_mSFS_*.pickle files in {simulations}.")
    arraydict = {}
    for path in paths:
        model_number = int(path.split('_')[-1].split('.')[0])
        with open(path, 'rb') as f:
            arraydict[model_number] = pickle.load(f)

    model_numbers = sorted(arraydict)
    label_to_int = {number: i for i, number in enumerate(model_numbers)}
    int_to_label = {i: number for number, i in label_to_int.items()}

    sfs = np.array([replicate for number in model_numbers for replicate in arraydict[number]])
    labels = [label_to_int[number] for number in model_numbers for _ in arraydict[number]]
    labels = keras.utils.to_categorical(labels, num_classes=len(model_numbers))

    return arraydict, sfs, labels, label_to_int, int_to_label, len(model_numbers)

def get_labels(dir:str, pattern:str):
    pattern_path = os.path.join(dir, pattern)
    paths = glob.glob(pattern_path)
    sorted_paths = sorted(paths, key=human_sort_key)
    labels = []

    for i in range(0, len(sorted_paths)):
        modno = sorted_paths[i].split('_')[-1].split('.')[0]
        labels.append(int(modno))
    return labels

def plot_pca(train_features, train_labels, new_features, pcafile):
    """
    Perform PCA and plot training vs. new feature distributions with a categorical legend.

    train_features: Extracted features from training data
    train_labels: Labels for training features
    new_features: Extracted features from new data
    pcafile: Path to save the PCA plot
    """

    # Perform PCA transformation (2 components)
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_features)
    new_pca = pca.transform(new_features)

    # Create a color map for the labels
    unique_labels = np.unique(train_labels)
    colors = plt.colormaps["Dark2"].resampled(len(unique_labels))  # Using the Dark2 color map
    
    # Plot training features
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        label_mask = train_labels == label
        plt.scatter(train_pca[label_mask, 0], train_pca[label_mask, 1], 
                    color=colors(i), label=str(label), alpha=0.7)

    # Plot new features (label as 'empirical' and red 'x' marker)
    plt.scatter(new_pca[:, 0], new_pca[:, 1], c="red", marker="x", alpha=0.7, label="New Data (Empirical)")

    # Add titles and labels
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Training Data Classes", loc='upper left')
    plt.title("PCA of Training and New Features")

    # Save the plot to the specified file
    plt.savefig(pcafile)
    plt.close()