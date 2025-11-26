from __future__ import annotations

import os
import pickle
import random
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import yaml
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Reshape,
    SimpleRNN,
)

from PNC_SYN.metrics_logger import CustomProgressBar, MetricsLogger
from PNC_SYN.postprocessing.log_postprocessing import generate_df
from PNC_SYN.preprocessing.log_preprocessing import END_TOKEN, START_TOKEN, preprocess_event_log
from PNC_SYN.preprocessing.log_tokenization import tokenize_log
from PNC_SYN.sampling.log_sampling import sample_batch


def _load_pickle_file(file_path: str) -> Any:
    """Load a trusted pickle artifact stored alongside the trained model."""
    with open(file_path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - local, versioned artifacts only

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]


class PNCEventLogSynthesizer:
    """Sequence model for event log synthesis with dual output heads.

    Builds and trains an autoregressive model that predicts the next event
    concept as well as the inter-event time. Provides utilities for
    preprocessing, tokenization, training with callbacks, and saving/loading
    all artifacts.

    Args:
        embedding_output_dims: Size of the embedding vectors.
        method: Encoder type ("LSTM", "Bi-LSTM", "GRU", "Bi-GRU", "RNN", "Bi-RNN").
        units_per_layer: Hidden units per recurrent layer.
        epochs: Default number of training epochs.
        batch_size: Training batch size.
        dropout: Dropout rate applied before the output layers.
        trace_quantile: Quantile used to bound trace length.
        learning_rate: Optimizer learning rate.
        validation_split: Fraction of training data used for validation.
        checkpoint_path: Optional path for training-time checkpoints.
        seed: Random seed for reproducibility. If None, a seed is generated.
    """

    def __init__(
        self,
        embedding_output_dims: int = 16,
        method: str = "LSTM",
        units_per_layer: list[int] | None = None,
        epochs: int = 3,
        batch_size: int = 16,
        dropout: float = 0.0,
        trace_quantile: float = 0.95,
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        checkpoint_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.modified_column_list: list[str] = []
        self.metrics_df: pd.DataFrame | None = None
        self.dict_dtypes: dict[str, Any] | None = None
        self.trace_quantile = trace_quantile

        self.model: Model | None = None
        self.max_sequence_len: int | None = None
        self.total_words: int = 0
        self.tokenizer: Any = None
        self.event_xs: IntArray | None = None
        self.event_ys: IntArray | None = None
        self.time_xs: FloatArray | None = None
        self.time_ys: FloatArray | None = None
        self.start_epoch: list[float] = []
        self.column_list: list[str] = ["concept:name", "time:timestamp"]
        self.num_cols: int = len(self.column_list)

        self.units_per_layer = list(units_per_layer) if units_per_layer else [64, 64]
        if not isinstance(self.units_per_layer, list) or not all(
            isinstance(u, int) and u > 0 for u in self.units_per_layer
        ):
            raise ValueError("units_per_layer must be a list of positive ints")

        allowed_methods = {"LSTM", "Bi-LSTM", "GRU", "Bi-GRU", "RNN", "Bi-RNN"}
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {sorted(allowed_methods)}")
        self.method = method
        self.embedding_output_dims = embedding_output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.checkpoint_path = checkpoint_path

        if seed is None:
            try:
                seed = random.SystemRandom().randint(0, 2**31 - 1)
            except Exception:
                seed = int.from_bytes(os.urandom(4), "little")
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.num_examples: int = 0

    def initialize_model(self, input_data: pd.DataFrame) -> None:  # noqa: C901 - orchestration heavy
        """Prepare data, build the network, and compile the model.

        Args:
            input_data: Raw event log DataFrame to preprocess and tokenize.
        """
        (
            event_sequences,
            time_sequences,
            self.dict_dtypes,
            self.start_epoch,
            self.num_examples,
        ) = preprocess_event_log(input_data, self.trace_quantile)

        (
            self.event_xs,
            self.event_ys,
            self.time_xs,
            self.time_ys,
            self.total_words,
            self.max_sequence_len,
            self.tokenizer,
        ) = tokenize_log(event_sequences, time_sequences)

        if self.event_xs is None or self.time_xs is None:
            raise RuntimeError("Tokenization failed to produce input tensors.")

        event_input = Input(shape=(self.max_sequence_len,), dtype="int32", name="event_tokens")
        time_input = Input(shape=(self.max_sequence_len,), dtype="float32", name="time_deltas")

        event_embedding = Embedding(
            self.total_words,
            self.embedding_output_dims,
            input_length=self.max_sequence_len,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
            mask_zero=True,
        )(event_input)
        time_expanded = Reshape((self.max_sequence_len, 1), name="expand_time")(time_input)
        x = Concatenate(axis=-1, name="fuse_event_time")([event_embedding, time_expanded])

        # Stack recurrent layers and pool the final timestep embedding
        for idx, units in enumerate(self.units_per_layer):
            is_last = idx == (len(self.units_per_layer) - 1)
            return_seq = not is_last
            if self.method == "LSTM":
                x = LSTM(units, return_sequences=return_seq)(x)
            elif self.method == "Bi-LSTM":
                x = Bidirectional(LSTM(units, return_sequences=return_seq))(x)
            elif self.method == "GRU":
                x = GRU(units, return_sequences=return_seq)(x)
            elif self.method == "Bi-GRU":
                x = Bidirectional(GRU(units, return_sequences=return_seq))(x)
            elif self.method == "RNN":
                x = SimpleRNN(units, return_sequences=return_seq)(x)
            elif self.method == "Bi-RNN":
                x = Bidirectional(SimpleRNN(units, return_sequences=return_seq))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        event_output = Dense(
            self.total_words, activation="softmax", name="concept_name"
        )(x)
        time_output = Dense(1, activation="sigmoid", name="time_timestamp")(x)

        self.modified_column_list = ["concept_name", "time_timestamp"]

        self.model = Model(inputs=[event_input, time_input], outputs=[event_output, time_output])
        optimizer = self._build_optimizer()
        self.model.compile(
            loss=["sparse_categorical_crossentropy", "mse"],
            optimizer=optimizer,
            metrics=["accuracy", "mae"],
        )

    def _build_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Return the standard Adam optimizer."""
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def train(self, epochs: int | None = None) -> None:
        """Train the model with early stopping, metrics logging, and optional checkpoints.

        Args:
            epochs: Number of epochs. Defaults to the value set at initialization.
        """
        if (
            self.model is None
            or self.event_xs is None
            or self.event_ys is None
            or self.time_xs is None
            or self.time_ys is None
        ):
            raise RuntimeError("Model must be initialized before training.")

        inputs = [self.event_xs, self.time_xs]
        targets = [self.event_ys, self.time_ys]

        monitor_metric = "val_concept_name_accuracy"
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            verbose=0,
            patience=7,
            restore_best_weights=True,
            min_delta=0.001,
            baseline=None,
            start_from_epoch=5,
        )

        metrics_logger = MetricsLogger(num_cols=self.num_cols, column_list=self.column_list)
        custom_progress_bar = CustomProgressBar()
        callbacks = [early_stopping, metrics_logger, custom_progress_bar]
        if self.checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    monitor=monitor_metric,
                    mode="max",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=0,
                )
            )

        self.model.fit(
            inputs,
            targets,
            epochs=epochs or self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            validation_split=self.validation_split,
            verbose=0,
        )

        self.metrics_df = metrics_logger.get_dataframe()

    def fit(self, input_data: pd.DataFrame) -> None:
        """Initialize the model and run training on the provided data.

        Args:
            input_data: Event log DataFrame.
        """
        self.initialize_model(input_data)
        self.train(self.epochs)

    def sample(self, sample_size: int, batch_size: int | None = None) -> pd.DataFrame:
        """Generate a synthetic event log using the trained model.

        Args:
            sample_size: Number of traces to sample.
            batch_size: Optional batch size for sampling.

        Returns:
            A DataFrame containing the sampled event log.
        """
        if (
            self.model is None
            or self.tokenizer is None
            or self.max_sequence_len is None
            or self.dict_dtypes is None
            or not self.start_epoch
        ):
            raise RuntimeError("Model must be trained or loaded before sampling.")

        total_sequences = 0
        synthetic_event_sequences: list[list[str]] = []
        synthetic_time_sequences: list[list[float]] = []
        batch = batch_size or self.batch_size

        while total_sequences < sample_size:
            remaining = sample_size - total_sequences
            events_batch, times_batch = sample_batch(
                remaining,
                self.tokenizer,
                self.max_sequence_len,
                self.model,
                batch,
                START_TOKEN,
                END_TOKEN,
            )
            synthetic_event_sequences.extend(events_batch)
            synthetic_time_sequences.extend(times_batch)
            total_sequences += len(events_batch)

        df = generate_df(
            synthetic_event_sequences,
            synthetic_time_sequences,
            self.dict_dtypes,
            self.start_epoch,
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def save_model(self, path: str) -> None:
        """Persist the model, checkpoints, metrics, and preprocessing artifacts.

        Args:
            path: Destination directory.
        """
        if self.model is None:
            raise RuntimeError("Train or load a model before saving.")
        if self.tokenizer is None or self.dict_dtypes is None:
            raise RuntimeError("Tokenizer and preprocessing artifacts must be available to save.")

        os.makedirs(path, exist_ok=True)

        self.model.save(os.path.join(path, "model.keras"))
        checkpoints_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        full_checkpoint_path = os.path.join(checkpoints_dir, "best.keras")
        self.model.save(full_checkpoint_path)
        if self.metrics_df is not None and not self.metrics_df.empty:
            try:
                self.metrics_df.to_excel(os.path.join(path, "training_metrics.xlsx"), index=False)
            except Exception:
                self.metrics_df.to_csv(os.path.join(path, "training_metrics.csv"), index=False)

        config = {
            "embedding_output_dims": self.embedding_output_dims,
            "method": self.method,
            "units_per_layer": self.units_per_layer,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "trace_quantile": self.trace_quantile,
            "num_examples": self.num_examples,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "checkpoint_path": os.path.join("checkpoints", "best.keras"),
            "seed": self.seed,
        }

        with open(os.path.join(path, "model_config.yaml"), "w", encoding="utf-8") as handle:
            yaml.dump(config, handle, default_flow_style=False)

        with open(os.path.join(path, "tokenizer.pkl"), "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "dict_dtypes.yaml"), "w", encoding="utf-8") as handle:
            yaml.dump(self.dict_dtypes, handle, default_flow_style=False)

        with open(os.path.join(path, "max_sequence_len.pkl"), "wb") as handle:
            pickle.dump(self.max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "start_epoch.pkl"), "wb") as handle:
            pickle.dump(self.start_epoch, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "num_cols.pkl"), "wb") as handle:
            pickle.dump(self.num_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, "column_list.pkl"), "wb") as handle:
            pickle.dump(self.column_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load a saved model and all required artifacts from a directory.

        Args:
            path: Directory containing the saved model and artifacts.
        """
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"), compile=False)

        self.tokenizer = _load_pickle_file(os.path.join(path, "tokenizer.pkl"))

        with open(os.path.join(path, "dict_dtypes.yaml"), encoding="utf-8") as handle:
            self.dict_dtypes = yaml.safe_load(handle)

        self.max_sequence_len = _load_pickle_file(os.path.join(path, "max_sequence_len.pkl"))
        self.start_epoch = _load_pickle_file(os.path.join(path, "start_epoch.pkl"))
        self.num_cols = _load_pickle_file(os.path.join(path, "num_cols.pkl"))
        self.column_list = _load_pickle_file(os.path.join(path, "column_list.pkl"))

        config_path = os.path.join(path, "model_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            self.embedding_output_dims = cfg.get(
                "embedding_output_dims", self.embedding_output_dims
            )
            self.method = cfg.get("method", self.method)
            self.units_per_layer = cfg.get("units_per_layer", self.units_per_layer)
            self.epochs = cfg.get("epochs", self.epochs)
            self.batch_size = cfg.get("batch_size", self.batch_size)
            self.dropout = cfg.get("dropout", self.dropout)
            self.trace_quantile = cfg.get("trace_quantile", self.trace_quantile)
            self.num_examples = cfg.get("num_examples", self.num_examples)
            self.learning_rate = cfg.get("learning_rate", self.learning_rate)
            self.validation_split = cfg.get("validation_split", self.validation_split)
            self.checkpoint_path = cfg.get("checkpoint_path", self.checkpoint_path)
            self.seed = cfg.get("seed", self.seed)

        self.modified_column_list = [
            c.replace(":", "_").replace(" ", "_") for c in (self.column_list or [])
        ]
