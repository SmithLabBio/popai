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
#from memory_profiler import profile
from popai.dataset import PopaiTrainingData
from typing import Dict
import glob
import re
from torch.utils.data import DataLoader
from tensorflow.keras.optimizers import Adam

class RandomForestsSFS:
    """Build a RF predictor that takes the SFS as input."""

    def __init__(self, config, simulations, user=False):
        self.config = config
        self.arraydict, self.sfs, self.labels, self.label_to_int, self.int_to_label, self.nclasses = read_data(simulations, user, type='1d')
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
        self.conv0_layers = []
        self.conv1_layers = []
        self.rows = []
        for key, num_rows in sampling_dict.items():
            self.rows.append(num_rows)
            conv_a_layer = keras.layers.Conv2D(10, (3, 1), strides=(1,1), activation="relu", padding="same")
            conv_layer = keras.layers.Conv2D(10, (int(num_rows/2), 1), strides=(int(num_rows/2), 1), 
                    activation="relu", padding="valid")
            self.conv0_layers.append(conv_a_layer)
            self.conv1_layers.append(conv_layer)
        self.conv2 = keras.layers.Conv2D(10, (len(sampling_dict), 1), activation="relu", 
                                         padding="valid")
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.drop = keras.layers.Dropout(0.1)
        self.dense2 = keras.layers.Dense(50, activation='relu')
        self.dense3 = keras.layers.Dense(n_classes, activation="softmax")

        self.pooling = keras.layers.AveragePooling2D(pool_size=(2, 1), strides=(2, 1), padding="valid")
        self.pool2 = keras.layers.AveragePooling2D(pool_size=(1,2), strides=(1,2), padding="valid")
        self.globalpool = keras.layers.GlobalAveragePooling2D()

    def call(self, x, return_intermediate=False):
        x = tf.cast(tf.expand_dims(x, axis=-1), dtype=tf.float64) # Reshape input and cast to float dtype
        outputs = []
        start_ix = 0
        for i in range(len(self.rows)):
            num_rows = self.rows[i]
            end_ix = start_ix + num_rows 
            out = self.conv0_layers[i](x[:,start_ix:end_ix,:,:])
            #print(f"Conv 0 2D layer {i} output shape:", out.shape)
            out = self.pooling(out)
            #print(f"Pool layer {i} output shape:", out.shape)
            out = self.conv1_layers[i](out)
            #print(f"Conv 1 2D layer {i} output shape:", out.shape)
            outputs.append(out)
        out = keras.layers.concatenate(outputs, axis=1)
        out = self.conv2(out)
        #print(f"Conv 2 2D layer output shape:", out.shape)
        out = self.pool2(out)
        #print(f"Pool layer output shape:", out.shape)
        #out = self.globalpool(out)
        #print(f"Global pooling output shape:", out.shape)
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

    if label=="cnn":
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        for epoch in range(epochs):
            for batch_data, batch_labels in batch_generator(data.train_loader):
                inputs = [tf.convert_to_tensor(pair, dtype=tf.float32) for pair in batch_data]
                labels = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

                model.train_on_batch(inputs, labels)

            for val_batch_data, val_batch_labels in batch_generator(data.test_loader):
                val_inputs = [tf.convert_to_tensor(pair, dtype=tf.float32) for pair in val_batch_data]
                val_labels = tf.convert_to_tensor(val_batch_labels, dtype=tf.float32)
                model.test_on_batch(val_inputs, val_labels)
        model.save(os.path.join(outdir, f"{label}.keras"))

    else:
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(data.train_loader, epochs=epochs, batch_size=batch_size, validation_data=data.test_loader)
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
    if label=="cnn":
        y_pred = []
        for batch_data, _ in data.test_loader:
            inputs = [tf.convert_to_tensor(pair, dtype=tf.float32) for pair in batch_data]  # Convert the population pair batch data
            y_hat = model.predict(inputs)  # Predict for the batch
            y_pred_batch = np.argmax(y_hat, axis=1).tolist()  # Convert to predicted labels
            y_pred.extend(y_pred_batch)
    else:
        y_hat = model.predict(data.test_loader)
        y_pred = np.argmax(y_hat, axis=1).tolist()
    cm, cm_plot = plot_confusion_matrix(y_true, y_pred, labels=[str(i) for i in y_true]) 
    cm_plot.savefig(os.path.join(outdir, f"{label}_confusion.png"))
    # Write to a TSV file
    with open(os.path.join(outdir, f"{label}_confusion.tsv"), "w") as f:
        # Write header
        f.write("\t" + "\t".join(map(str, y_true)) + "\n")
        # Write rows
        for i, row in zip(y_true, cm):
            f.write(str(i) + "\t" + "\t".join(map(str, row)) + "\n")

def extract_features(model, dataset, label):
    """
    Extract features from a dataset using the trained model.

    model: Trained Keras model
    dataset: DataLoader or other iterable dataset

    Returns:
        - features (numpy array)
        - labels (numpy array) (if available, otherwise None)
    """
    if isinstance(dataset, DataLoader):
        extracted_features = []
        all_labels = []

        if label=="cnn": # TODO: FIX THIS
            for x_batch, y_batch in dataset:
                features = model(x_batch, return_intermediate=True)
                extracted_features.append(features.numpy())
                all_labels.append(y_batch)
        else:
            for x_batch, y_batch in dataset:
                features = model(x_batch, return_intermediate=True)
                extracted_features.append(features.numpy())
                all_labels.append(y_batch)
        extracted_features = np.concatenate(extracted_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return extracted_features, all_labels
    else:
        if label=="cnn":
            extracted_features = model(dataset, return_intermediate=True)
        
        else:
            extracted_features = model(dataset, return_intermediate=True)
        return extracted_features.numpy()

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

    if label=="cnn":

        pred = []
        for batch_data in data:
            inputs = [tf.convert_to_tensor(pair, dtype=tf.float32) for pair in batch_data]
            for item in range(len(inputs)):
                inputs[item] =  inputs[item] / np.sum(inputs[item])  # Normalize so that elements sum to 1
            inputs = [tf.expand_dims(pair, axis=0) for pair in inputs]
            pred.extend(model.predict(inputs))
        replicate_numbers = ["Replicate {}".format(i+1) for i in range(len(pred))]



    else:
        pred = model.predict(data)
        replicate_numbers = ["Replicate {}".format(i+1) for i in range(pred.shape[0])]


    labels = get_labels(sim_dir, path)
    headers = [f"Model {i}" for i in labels]
    table_data = np.column_stack((replicate_numbers, pred))
    tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

    with open(os.path.join(out_dir, f"{label}_predictions.txt"), 'w') as fh:
        fh.write(tabulated)

    # feature extraction
    if label=="cnn":
        new_features = []
        for batch_data in data:
            inputs = [tf.convert_to_tensor(pair, dtype=tf.float32) for pair in batch_data]
            for item in range(len(inputs)):
                inputs[item] =  inputs[item] / np.sum(inputs[item])  # Normalize so that elements sum to 1
            inputs = [tf.expand_dims(pair, axis=0) for pair in inputs]
            new_features.append(extract_features(model, inputs, label))
        new_features = np.concatenate(new_features, axis=0)
    else:
        new_features= extract_features(model, data, label)
    new_features_path = os.path.join(out_dir, f"{label}_empirical_features.npy")
    np.save(new_features_path, new_features)

    # load training
    train_features = np.load(train_features_path)
    train_labels = np.load(train_labels_path)

    # pca
    outputpca = os.path.join(out_dir, f"{label}_pca.png")
    plot_pca(train_features, np.argmax(train_labels, axis=1), new_features, outputpca)

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

def read_data(simulations, user, type):

    arraydict = {}
    sfs = []
    labels = []

    # read in the data
    pickle_list = os.listdir(simulations)
    if type=='1d':
        pickle_list = [x for x in pickle_list if ('_mSFS' in x and x.endswith('.pickle'))]
    elif type=='2d':
        pickle_list = [x for x in pickle_list if ('_2dSFS' in x and x.endswith('.pickle'))]
    elif type=='npy':
        pickle_list = [x for x in pickle_list if ('_arrays' in x and x.endswith('.pickle'))]
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
    colors = plt.cm.get_cmap("Dark2", len(unique_labels))  # Using the Dark2 color map

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

def batch_generator(loader):
    for batch_data, batch_labels in loader:
        yield (batch_data, batch_labels)



def human_sort_key(s):
    """
    Key function for human sorting.
    Splits the string into parts and converts numeric parts to integers.
    """
    return [int(part) if part.isdigit() else part for part in re.split('([0-9]+)', s)]