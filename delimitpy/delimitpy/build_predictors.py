"""Build predictive models."""
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class RandomForestsSFS:

    """Build a RF predictor that takes the SFS as input."""

    def __init__(self, config, sfs, labels):
        self.config = config
        self.sfs = [item for sublist in sfs for item in sublist]
        self.sfs = np.array(self.sfs)
        self.labels = [item for sublist in labels for item in sublist]
        self.rng = np.random.default_rng(self.config['seed'])

    def build_rf_sfs(self):

        """Build a random forest classifier that takes the
        multidimensional SFS as input."""
        
        train_test_seed = self.rng.integers(2**32, size=1)[0]

        x_train, x_test, y_train, y_test = train_test_split(self.sfs,
                self.labels, test_size=0.2, random_state=train_test_seed)

        sfs_rf = RandomForestClassifier(n_estimators=100, oob_score=True)

        sfs_rf.fit(x_train, y_train)
        print("Out-of-Bag (OOB) Error:", 1.0 - sfs_rf.oob_score_)


        cv_scores = cross_val_score(sfs_rf, x_test, y_test, cv=2)
        print("Cross-validation scores:", cv_scores)

        y_pred_cv = cross_val_predict(sfs_rf, x_test, y_test, cv=2)
        conf_matrix = confusion_matrix(y_test, y_pred_cv)
        conf_matrix_plot = plot_confusion_matrix(y_test, y_pred_cv)


        return sfs_rf, conf_matrix, conf_matrix_plot

    def predict(self, model, new_data):
        new_data = np.array(new_data)
        predicted = model.predict(new_data)
        predicted_prob = model.predict_proba(new_data)
        headers = ["Model {}".format(i) for i in range(predicted_prob.shape[1])]
        replicate_numbers = ["Replicate {}".format(i+1) for i in range(predicted_prob.shape[0])]
        table_data = np.column_stack((replicate_numbers, predicted_prob))
        tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        return(tabulated)

class CnnSFS:

    """Build a CNN predictor that takes the SFS as input."""

    def __init__(self, config, sfs_2d, labels):
        self.config = config
        self.sfs_2d = [item for sublist in sfs_2d for item in sublist]
        self.nclasses = len(labels)
        self.labels = np.array([item for sublist in labels for item in sublist])
        self.rng = np.random.default_rng(self.config['seed'])
        try:
            self.labels = keras.utils.to_categorical(self.labels)
        except:
            pass

    def build_cnn_sfs(self):

        """Build a CNN that takes 2D SFS as input."""

        # get features
        list_of_features = self._convert_2d_dictionary(self.sfs_2d)

        # shuffle data
        num_samples = len(self.labels)

        # Create an array of indices and shuffle it
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the shuffled indices into training and validation indices
        split_ratio = 0.8  # 80% training, 20% validation
        split_idx = int(num_samples * split_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Split features and labels into training and validation sets using the indices
        train_features = [[list_of_features[j][i] for i in train_indices]\
                          for j in range(len(list_of_features))]
        val_features = [[list_of_features[j][i] for i in val_indices]\
                        for j in range(len(list_of_features))]
        train_labels = self.labels[train_indices]
        val_labels = self.labels[val_indices]

        # to arrays
        train_features = [np.expand_dims(np.array(x), axis=-1) for x in train_features]
        val_features = [np.expand_dims(np.array(x), axis=-1) for x in val_features]

        # build model
        my_layers = []
        inputs = []
        for item in train_features:
            this_input = keras.Input(shape=item.shape[1:])
            x =  keras.layers.Conv2D(10, (4,4), activation="relu")(this_input)
            x = keras.layers.Flatten()(x)
            my_layers.append(x)
            inputs.append(this_input)

        concatenated = keras.layers.Concatenate()(my_layers)
        x = keras.layers.Dense(64, activation='relu')(concatenated)
        x = keras.layers.Dense(self.nclasses, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_features, train_labels, epochs=10,
                  batch_size=10, validation_data=(val_features, val_labels))

        val_pred = model.predict(val_features)
        val_predicted_labels = np.argmax(val_pred, axis=1)
        val_true_labels = np.argmax(val_labels, axis=1)
        conf_matrix = confusion_matrix(val_true_labels, val_predicted_labels)
        conf_matrix_plot = plot_confusion_matrix(val_true_labels, val_predicted_labels)

        return model, conf_matrix, conf_matrix_plot

    def predict(self, model, new_data):
        new_features = self._convert_2d_dictionary(new_data)
        new_features = [np.expand_dims(np.array(x), axis=-1) for x in new_features]
        predicted = model.predict(new_features)
        headers = ["Model {}".format(i) for i in range(predicted.shape[1])]
        replicate_numbers = ["Replicate {}".format(i+1) for i in range(predicted.shape[0])]
        table_data = np.column_stack((replicate_numbers, predicted))
        tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

        return(tabulated)

    def _convert_2d_dictionary(self, data):

        list_of_features = []

        for data_dict in data:

            count = 0

            for value in data_dict.values():

                if len(list_of_features) >= count+1:
                    list_of_features[count].append(np.array(value))
                else:
                    list_of_features.append([np.array(value)])

                count+=1

        return list_of_features

class NeuralNetSFS:

    """Build a RF predictor that takes the SFS as input."""

    def __init__(self, config, sfs, labels):
        self.config = config
        self.sfs = [item for sublist in sfs for item in sublist]
        self.sfs = np.array(self.sfs)
        self.nclasses = len(labels)
        self.labels = np.array([item for sublist in labels for item in sublist])
        self.rng = np.random.default_rng(self.config['seed'])
        try:
            self.labels = keras.utils.to_categorical(self.labels)
        except:
            pass


    def build_neuralnet_sfs(self):

        """Build a neural network classifier that takes the
        multidimensional SFS as input."""

        # split train and test
        train_test_seed = self.rng.integers(2**32, size=1)[0]

        x_train, x_test, y_train, y_test = train_test_split(self.sfs,
                self.labels, test_size=0.2, random_state=train_test_seed)

        # build model
        network_input = keras.Input(shape=x_train.shape[1:])
        x = keras.layers.Dense(100, activation='relu')(network_input)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dense(self.nclasses, activation='softmax')(x)

        # fit model
        model = keras.Model(inputs=network_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10,
                  batch_size=10, validation_data=(x_test, y_test))

        # evaluate model
        val_pred = model.predict(x_test)
        val_predicted_labels = np.argmax(val_pred, axis=1)
        val_true_labels = np.argmax(y_test, axis=1)
        conf_matrix = confusion_matrix(val_true_labels, val_predicted_labels)
        conf_matrix_plot = plot_confusion_matrix(val_true_labels, val_predicted_labels)

        return model, conf_matrix, conf_matrix_plot
    
    def predict(self, model, new_data):

        new_data = np.array(new_data)
        predicted = model.predict(new_data)
        predicted = model.predict(new_data)
        headers = ["Model {}".format(i) for i in range(predicted.shape[1])]
        replicate_numbers = ["Replicate {}".format(i+1) for i in range(predicted.shape[0])]
        table_data = np.column_stack((replicate_numbers, predicted))
        tabulated = tabulate(table_data, headers=headers, tablefmt="fancy_grid")

        return(tabulated)

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    return plt