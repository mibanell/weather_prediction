import numpy as np
import pandas as pd
import datetime

class WeatherPreprocessing():

    def __init__(self):
        self.data = None
        self.means = None
        self.stds = None

    def clean(self):

        # slice [start:stop:step], starting from index 5 take every 6th record, to take one sample per hour
        self.data = self.data.copy().loc[5::6]

        self.date_time = pd.to_datetime(self.data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

        # Remove incorrect data ---------------
        wv = self.data['wv (m/s)']
        bad_wv = wv == -9999.0
        wv[bad_wv] = 0.0

        max_wv = self.data['max. wv (m/s)']
        bad_max_wv = max_wv == -9999.0
        max_wv[bad_max_wv] = 0.0

    def feature_engineer(self):
        # Create direction vector combining direction degree and speed
        wv = self.data.pop('wv (m/s)')
        max_wv = self.data.pop('max. wv (m/s)')

        # Convert to radians.
        wd_rad = self.data.pop('wd (deg)') * np.pi / 180

        # Calculate the wind x and y components.
        self.data['Wx'] = wv * np.cos(wd_rad)
        self.data['Wy'] = wv * np.sin(wd_rad)

        # Calculate the max wind x and y components.
        self.data['max Wx'] = max_wv * np.cos(wd_rad)
        self.data['max Wy'] = max_wv * np.sin(wd_rad)

        # Set "hour of day " and "hour of year" features using sin and cos
        timestamp_s = self.date_time.map(datetime.datetime.timestamp)

        day = 24 * 60 * 60
        year = (365.2425) * day

        self.data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    def normalize(self, calculate_metrics):
        if calculate_metrics:
            self.means = self.data.mean()
            self.stds = self.data.std()
        else:
            self.means = self.means[self.data.columns]
            self.stds = self.stds[self.data.columns]

        self.data = (self.data - self.means) / self.stds

    def fit(self, data):
        self.data = data
        # Clean data
        self.clean()
        # Feature engineering
        self.feat_eng()
        # Normalization
        self.normalize(calculate_metrics=True)

    def fit_transform(self, data):
        self.data = data
        # Clean data
        self.clean()
        # Feature engineering
        self.feat_eng()
        # Normalization
        self.normalize(calculate_metrics=True)

        return self.data

    def transform(self, data):
        self.data = data
        # Clean data
        self.clean()
        # Feature engineering
        self.feat_eng()
        # Normalization
        self.normalize(calculate_metrics=False)

        return self.data

    def save_params(self, path):
        # Create dataframe with means and standard deviations from training
        params = pd.DataFrame({'feature': self.means.index, 'mean': self.means.values, 'std': self.stds.values})
        # Save as CSV
        params.to_csv(path, index=False)
        if 'google.colab' in str(get_ipython()):
            files.download(path)

        def load_params(self, path):
            params = pd.read_csv(path)

            means = params['mean']
            means.index = params['feature']
            self.means = params['mean']

            stds = params['std']
            stds.index = params['feature']
            self.stds = params['std']



class WindowGenerator():

    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
          labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
          plt.subplot(3, 1, n+1)
          plt.ylabel(f'{plot_col} [normed]')
          plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                  label='Inputs', marker='.', zorder=-10)

          if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
          else:
            label_col_index = plot_col_index

          if label_col_index is None:
            continue

          plt.scatter(self.label_indices, labels[n, :, label_col_index],
                      edgecolors='k', label='Labels', c='#2ca02c', s=64)
          if model is not None:
            predictions = model.model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, :],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

          if n == 0:
            plt.legend()

          plt.xlabel('Time [h]')

    def make_dataset(self, data, shuffle):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self, shuffle=True):
        return self.make_dataset(self.train_df, shuffle=True)

    @property
    def train_no_shuffle(self):
        return self.make_dataset(self.train_df, shuffle=False)

    @property
    def val(self):
        return self.make_dataset(self.val_df, shuffle=False)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
          # No example batch was found, so get one from the `.train` dataset
          result = next(iter(self.train))
          # And cache it for next time
          self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


class WeatherModel():

    def __init__(self, input_shape):
        # Initialize model
        # Input layer (batch, timesteps, features)
        inputs = tf.keras.layers.Input(shape=input_shape)
        # LSTM layer, 12 units
        lstm = tf.keras.layers.LSTM(12)(inputs)
        # Dense layer
        dense = tf.keras.layers.Dense(
            OUT_STEPS * 1,
            activation='linear',
            kernel_initializer=tf.initializers.zeros
        )(lstm)
        # Shape => [batch, out_steps, label_features]
        reshape = tf.keras.layers.Reshape([OUT_STEPS, 1])(dense)

        multi_lstm_model = tf.keras.Model(inputs=inputs, outputs=reshape)

        self.model = multi_lstm_model

    def summary(self):
        return self.model.summary()

    def compile_and_fit(self, window, patience, max_epochs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        self.history = self.model.fit(window.train, epochs=max_epochs,
                                      validation_data=window.val,
                                      callbacks=[early_stopping])

    def predict(self, window):
        return self.model.predict(window)

    def save_weights(self, **kwargs):
        self.model.save_weights(**kwargs)
        if 'google.colab' in str(get_ipython()):
            files.download(kwargs['filepath'])

    def load_weights(self, **kwargs):
        self.model.load_weights(**kwargs)

    def visualize_loss(self):
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title('Loss evolution during training')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()