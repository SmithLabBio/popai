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
import glob
from torch.utils.data import Subset, DataLoader 
from memory_profiler import profile
from .dataset import PopaiDataset, PopaiDatasetLowMem

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


class CnnSFSModel(keras.Model):
    def __init__(self, pop_pairs, n_classes, name=None):
        super().__init__(name=name)
        self.pop_pairs = pop_pairs
        self.conv1_layers = [] 
        for i in pop_pairs:
            conv_layer = keras.layers.Conv2D(10, (3,3), activation="relu")
            self.conv1_layers.append(conv_layer)
        self.dense1 = keras.layers.Dense(64, activation="relu")
        self.dense2 = keras.layers.Dense(n_classes, activation="softmax")

    def call(self, x):
        outputs = []
        for pair, conv in zip(self.pop_pairs, self.conv1_layers):
            out = conv(tf.expand_dims(x[pair], axis=-1))
            out = keras.layers.Flatten()(out)
            outputs.append(out)
        out = keras.layers.concatenate(outputs)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

class CnnSFS:
    """Build a CNN predictor that takes the 2D SFS as input."""

    def __init__(self, config, simulations, low_mem=True):
        self.config = config
        self.rng = np.random.default_rng(self.config['seed'])
        if low_mem:
            self.dataset = PopaiDatasetLowMem(simulations)  
        else:
            self.dataset = PopaiDataset(simulations)  
    

    def build_cnn_sfs(self):
        """Build a CNN that takes 2D SFS as input."""
        
        # split train and test
        train_test_seed = self.rng.integers(2**32, size=1)[0]
        train_ixs, test_ixs = train_test_split(np.arange(len(self.dataset)), test_size=0.2, 
                random_state=train_test_seed, stratify=self.dataset.labels)
        train_dataset = Subset(self.dataset, train_ixs)
        test_dataset = Subset(self.dataset, train_ixs)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        # Define and train model
        pop_pairs = list(self.dataset[0][0].keys())
        self.model = CnnSFSModel(pop_pairs, self.dataset.n_classes)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(train_loader, epochs=10, batch_size=10, validation_data=test_loader) 

        # evaluate model
        y_test_pred = self.model.predict(test_loader)
        y_test_original = [self.dataset.labels[i] for i in test_dataset.indices]
        y_pred_original = np.argmax(y_test_pred, axis=1).tolist()

        self.conf_matrix, self.conf_matrix_plot = plot_confusion_matrix(y_test_original, y_pred_original, 
                labels=[str(i) for i in y_test_original])
        
        # extract the features
        self.feature_extractor = keras.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)


    # def predict(self, model, new_data):
    #     new_features = self._convert_2d_dictionary(new_data)
    #     new_features = [np.expand_dims(np.array(x), axis=-1) for x in new_features]
    #     predicted = model.predict(new_features)

    #     if predicted.shape[1] != self.nclasses:
    #         raise ValueError(f"Model has {predicted.shape[1]} classes, but the provided data has {self.nclasses} classes. You probably used different subsets for training and applying.")

    #     headers = [f"Model {self.int_to_label[i]}" for i in range(self.labels.shape[1])]
    #     replicate_numbers = ["Replicate {}".format(i+1) for i in range(predicted.shape[0])]
    #     table_data = np.column_stack((replicate_numbers, predicted))
    #     tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

    #     return(tabulated)

    # def check_fit(self, feature_extractor, new_data, output_directory):

    #     # features from empirical data
    #     new_features = self._convert_2d_dictionary(new_data)
    #     new_features = [np.expand_dims(np.array(x), axis=-1) for x in new_features]
    #     new_extracted_features = feature_extractor.predict(new_features)

    #     # features from training data
    #     list_of_features = self._convert_2d_dictionary(self.sfs_2d)
    #     train_features = [np.expand_dims(np.array(x), axis=-1) for x in list_of_features]
    #     train_extracted_features = feature_extractor.predict(train_features)

    #     # pca
    #     pca = PCA(n_components=2)
    #     train_pca = pca.fit_transform(train_extracted_features)
    #     new_pca = pca.transform(new_extracted_features)

    #     # plot
    #     training_labels = tf.argmax(self.labels, axis=1)
    #     unique_labels = np.unique(training_labels)
    #     for label in unique_labels:
    #         indices = np.where(np.array(training_labels) == label)
    #         plt.scatter(train_pca[indices, 0], train_pca[indices, 1], label=f"Train: {self.int_to_label[label]}")

    #     plt.scatter(new_pca[:, 0], new_pca[:, 1], color='black', label='New Data', marker='x')

    #     plt.xlabel('PCA 1')
    #     plt.ylabel('PCA 2')
    #     plt.legend()
        
    #     # Save the plot to the specified file
    #     plt.savefig(os.path.join(output_directory, 'cnn_2dsfs_features.png'), dpi=300, bbox_inches='tight')
    #     plt.close()  # Close the plot to avoid displaying it in interactive environments





class NeuralNetSFS:
    """Build a neural network predictor that takes the SFS as input."""

    def __init__(self, config, simulations, low_mem=True):
        self.config = config
        self.rng = np.random.default_rng(self.config['seed'])
        if low_mem:
            self.dataset = PopaiDatasetLowMem(simulations)  
        else:
            self.dataset = PopaiDataset(simulations)  

    def build_neuralnet_sfs(self):
        """Build a neural network classifier that takes the
        multidimensional SFS as input."""

        # split train and test
        train_test_seed = self.rng.integers(2**32, size=1)[0]
        train_ixs, test_ixs = train_test_split(np.arange(len(self.dataset)), test_size=0.2, 
                random_state=train_test_seed, stratify=self.dataset.labels)
        train_dataset = Subset(self.dataset, train_ixs)
        test_dataset = Subset(self.dataset, train_ixs)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        # build model
        network_input = keras.Input(shape=self.dataset[0][0].shape)
        x = keras.layers.Dense(100, activation='relu')(network_input)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dense(self.dataset.n_classes, activation='softmax')(x)

        # fit model
        model = keras.Model(inputs=network_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_loader, epochs=10, batch_size=10, validation_data=test_loader)

        # evaluate model
        y_test_pred = model.predict(test_loader)
        y_test_original = [self.dataset.labels[i] for i in test_dataset.indices]
        y_pred_original = np.argmax(y_test_pred, axis=1).tolist()

        conf_matrix, conf_matrix_plot = plot_confusion_matrix(y_test_original, y_pred_original, 
                labels=[str(i) for i in y_test_original])

        # extract the features
        feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

        return model, conf_matrix, conf_matrix_plot, feature_extractor

    # TODO: Fix this stuff 
    # def predict(self, model, new_data):

    #     new_data = np.array(new_data)
    #     predicted = model.predict(new_data)

    #     if predicted.shape[1] != self.nclasses:
    #         raise ValueError(f"Model has {predicted.shape[1]} classes, but the provided data has {self.nclasses} classes. You probably used different subsets for training and applying.")

    #     headers = [f"Model {self.int_to_label[i]}" for i in range(self.labels.shape[1])]
    #     replicate_numbers = ["Replicate {}".format(i+1) for i in range(predicted.shape[0])]
    #     table_data = np.column_stack((replicate_numbers, predicted))
    #     tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

    #     return(tabulated)

    # def check_fit(self, feature_extractor, new_data, output_directory):

    #     # features from empirical data
    #     new_data = np.array(new_data)
    #     new_extracted_features = feature_extractor.predict(new_data)
    #     train_extracted_features = feature_extractor.predict(np.array(self.sfs))

    #     # pca
    #     pca = PCA(n_components=2)
    #     train_pca = pca.fit_transform(train_extracted_features)
    #     new_pca = pca.transform(new_extracted_features)

    #     # plot
    #     training_labels = tf.argmax(self.labels, axis=1)
    #     unique_labels = np.unique(training_labels)
    #     for label in unique_labels:
    #         indices = np.where(np.array(training_labels) == label)
    #         plt.scatter(train_pca[indices, 0], train_pca[indices, 1], label=f"Train: {self.int_to_label[label]}")

    #     plt.scatter(new_pca[:, 0], new_pca[:, 1], color='black', label='New Data', marker='x')

    #     plt.xlabel('PCA 1')
    #     plt.ylabel('PCA 2')
    #     plt.legend()
        
    #     # Save the plot to the specified file
    #     plt.savefig(os.path.join(output_directory, 'fcnn_features.png'), dpi=300, bbox_inches='tight')
    #     plt.close()  # Close the plot to avoid displaying it in interactive environments


class CnnNpyModel(keras.Model):
    def __init__(self, n_sites, downsampling_dict, n_classes, name=None):
        super().__init__(name=name)
        self.conv1_layers = []
        self.rows = []
        for key, num_rows in downsampling_dict.items():
            self.rows.append(num_rows)
            conv_layer = keras.layers.Conv2D(10, (num_rows, 1), strides=(num_rows, 1), 
                    activation="relu", padding="valid") #(input_layer)
            self.conv1_layers.append(conv_layer)
        self.conv2 = keras.layers.Conv2D(10, (len(downsampling_dict), 1), activation="relu", padding="valid")
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.drop = keras.layers.Dropout(0.1)
        self.dense2 = keras.layers.Dense(50, activation='relu')
        self.dense3 = keras.layers.Dense(n_classes, activation="softmax")
    
    def call(self, x):
        x = tf.cast(tf.expand_dims(x, axis=-1), dtype=tf.float64) # Reshape input and cast to float
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

class CnnNpy:
    """Build a CNN predictor that takes the alignment as a numpy matrix as input."""

    def __init__(self, config, downsampling_dict, simulations, subset, user=False, low_mem=True):
        self.config = config
        self.arraydicts = {}
        self.arrays = []
        self.labels = []
        self.input = input
        # self.arraydict, self.arrays, self.labels, self.label_to_int, self.int_to_label, self.nclasses = read_data(simulations, subset, user, type='npy')
        self.rng = np.random.default_rng(self.config['seed'])
        self.downsampling_dict = {}
        for key,value in self.config['sampling dict'].items():
            self.downsampling_dict[key] = downsampling_dict[key]
        model_paths = glob.glob(f"{os.path.join(simulations, 'simulated_arrays_')}*.pickle") # TODO: Move this out of the class
        if low_mem:
            self.dataset = PopaiDatasetLowMem(model_paths)
        else:
            self.dataset = PopaiDataset(model_paths)
        
    def build_cnn_npy(self):
        """Build a CNN that takes npy array as input."""

        train_test_seed = self.rng.integers(2**32, size=1)[0]
        train_ixs, test_ixs = train_test_split(np.arange(len(self.dataset)), test_size=0.2, 
                random_state=train_test_seed, stratify=self.dataset.labels)
        train_dataset = Subset(self.dataset, train_ixs)
        test_dataset = Subset(self.dataset, train_ixs)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        # Define and train model
        model = CnnNpyModel(train_dataset[0][0].shape[1], self.downsampling_dict, 
                self.dataset.n_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_loader, epochs=10, batch_size=10,) 
                #   validation_data=test_loader) # TODO: Implement validation

        # evaluate model
        y_test_pred = model.predict(test_loader)
        y_test_original = [self.dataset.labels[i] for i in test_dataset.indices]
        y_pred_original = np.argmax(y_test_pred, axis=1).tolist()

        conf_matrix, conf_matrix_plot = plot_confusion_matrix(y_test_original, y_pred_original, 
                labels=[str(i) for i in y_test_original])

        # extract the features
        feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

        return model, conf_matrix, conf_matrix_plot, feature_extractor

    # TODO: Fix this stuff
    # def predict(self, model, new_data):
 
    #     new_data = np.expand_dims(new_data, axis=-1)
    #     new_data = np.expand_dims(new_data, axis=0)


    #     # split by pop
    #     split_features = []
    #     start_idx = 0
    #     for key, num_rows in self.downsampling_dict.items():
    #         end_idx = start_idx + num_rows
    #         split_features.append(new_data[:,start_idx:end_idx,:,:])
    #         start_idx = end_idx


    #     predicted = model.predict(split_features)
    #     if predicted.shape[1] != self.nclasses:
    #         raise ValueError(f"Model has {predicted.shape[1]} classes, but the provided data has {self.nclasses} classes. You probably used different subsets for training and applying.")
    #     headers = [f"Model {self.int_to_label[i]}" for i in range(self.labels.shape[1])]
    #     replicate_numbers = ["Replicate {}".format(i+1) for i in range(predicted.shape[0])]
    #     table_data = np.column_stack((replicate_numbers, predicted))
    #     tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

    #     return(tabulated)

    # def check_fit(self, feature_extractor, new_data, output_directory):

    #     # features from empirical data
    #     new_data = np.expand_dims(new_data, axis=-1)
    #     new_data = np.expand_dims(new_data, axis=0)
    #     training_data = np.array(self.arrays)
    #     training_data = np.expand_dims(training_data, axis=-1)

    #     # split by pop
    #     split_features = []
    #     split_train_features = []

    #     start_idx = 0
    #     for key, num_rows in self.downsampling_dict.items():
    #         end_idx = start_idx + num_rows
    #         split_features.append(new_data[:,start_idx:end_idx,:,:])
    #         split_train_features.append(training_data[:,start_idx:end_idx,:,:])
    #         start_idx = end_idx
    #     new_extracted_features = feature_extractor.predict(split_features)
    #     train_extracted_features = feature_extractor.predict(split_train_features)

    #     # pca
    #     pca = PCA(n_components=2)
    #     train_pca = pca.fit_transform(train_extracted_features)
    #     new_pca = pca.transform(new_extracted_features)

    #     # plot
    #     training_labels = tf.argmax(self.labels, axis=1)
    #     unique_labels = np.unique(training_labels)
    #     for label in unique_labels:
    #         indices = np.where(np.array(training_labels) == label)
    #         plt.scatter(train_pca[indices, 0], train_pca[indices, 1], label=f"Train: {self.int_to_label[label]}")

    #     plt.scatter(new_pca[:, 0], new_pca[:, 1], color='black', label='New Data', marker='x')

    #     plt.xlabel('PCA 1')
    #     plt.ylabel('PCA 2')
    #     plt.legend()
        
    #     # Save the plot to the specified file
    #     plt.savefig(os.path.join(output_directory, 'cnn_npy_features.png'), dpi=300, bbox_inches='tight')
    #     plt.close()  # Close the plot to avoid displaying it in interactive environments


def plot_confusion_matrix(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
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
            with open(os.path.join(simulations, 'simulated_mSFS_%s.pickle' % str(modno)), 'rb') as f:
                arraydict[str(modno)] = pickle.load(f)
        elif type=='2d':
            with open(os.path.join(simulations, 'simulated_2dSFS_%s.pickle' % str(modno)), 'rb') as f:
                arraydict[str(modno)] = pickle.load(f)
        elif type=='npy':
            with open(os.path.join(simulations, 'simulated_arrays_%s.pickle' % str(modno)), 'rb') as f:
                arraydict[str(modno)] = pickle.load(f)

    for key,value in arraydict.items():
        for thearray in value:
            sfs.append(thearray)
            labels.append(key)
    nclasses = len(set(labels))
    if user:
        try:
            labels = [int(x.split('_')[-1]) for x in labels]
            valid = check_valid_labels(labels)
        except:
            raise ValueError(f"Model names must be 'Model_x', where x are integers ranging from 0 to n-1, where n is the number of models.")
        if not valid:
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
