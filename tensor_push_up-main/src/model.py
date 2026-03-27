"""
Model Definition Module

This module defines deep learning models for action recognition using
LSTM and MLP architectures for temporal sequence classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _materialize_model(model: keras.Model, input_shape: Tuple[int, int]) -> None:
    """
    Run a dummy forward pass so subclassed models build their weights eagerly.

    This avoids misleading "0 params (unbuilt)" summaries and makes shape
    mismatches surface closer to model creation time.
    """
    dummy_input = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)
    model(dummy_input, training=False)


@keras.utils.register_keras_serializable(package="tensor_push_up")
class ActionClassifier(keras.Model):
    """
    LSTM-based action classifier for temporal sequence classification.

    Architecture:
        Input (batch, window_size, features)
            ↓
        LSTM Layer 1 (128 units)
            ↓
        Dropout (0.5)
            ↓
        LSTM Layer 2 (64 units)
            ↓
        Dropout (0.5)
            ↓
        Dense Layer 1 (32 units, ReLU)
            ↓
        Dense Layer 2 (16 units, ReLU)
            ↓
        Output Layer (num_classes, Softmax)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        lstm_units: List[int] = [128, 64],
        dense_units: List[int] = [32, 16],
        dropout_rate: float = 0.5,
        l2_reg: float = 0.001,
        **kwargs
    ):
        """
        Initialize the Action Classifier model.

        Args:
            input_shape: Input shape (window_size, features)
            num_classes: Number of output classes
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
        """
        kwargs.setdefault("name", "action_classifier")
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Build model layers
        self.lstm_layers = []
        self.dropout_layers = []

        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            self.lstm_layers.append(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    recurrent_regularizer=keras.regularizers.l2(l2_reg),
                    name=f"lstm_{i+1}"
                )
            )
            if dropout_rate > 0:
                self.dropout_layers.append(
                    layers.Dropout(dropout_rate, name=f"dropout_{i+1}")
                )

        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name=f"dense_{i+1}"
                )
            )

        # Output layer
        self.output_layer = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )

        _materialize_model(self, input_shape)
        logger.info(f"ActionClassifier built with input shape {input_shape}")

    def call(self, inputs, training=False):
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor (batch, window_size, features)
            training: Whether in training mode

        Returns:
            Output logits (batch, num_classes)
        """
        x = inputs

        # LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x = lstm(x)
            if training and i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)

        # Dense layers
        for dense in self.dense_layers:
            x = dense(x)

        # Output
        outputs = self.output_layer(x)

        return outputs

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config

    def summary(self, **kwargs):
        """Print model summary."""
        if not self.built:
            self.build((None,) + self.input_shape)
        super().summary(**kwargs)


@keras.utils.register_keras_serializable(package="tensor_push_up")
class BidirectionalActionClassifier(keras.Model):
    """
    Bidirectional LSTM action classifier for enhanced temporal modeling.

    Uses bidirectional LSTM layers to capture both forward and backward
    temporal dependencies in the action sequence.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        lstm_units: List[int] = [64, 32],
        dense_units: List[int] = [32],
        dropout_rate: float = 0.5,
        l2_reg: float = 0.001,
        **kwargs
    ):
        """
        Initialize the Bidirectional Action Classifier.

        Args:
            input_shape: Input shape (window_size, features)
            num_classes: Number of output classes
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
        """
        kwargs.setdefault("name", "bidirectional_action_classifier")
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Bidirectional LSTM layers
        self.bilstm_layers = []
        self.dropout_layers = []

        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            self.bilstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        recurrent_regularizer=keras.regularizers.l2(l2_reg)
                    ),
                    name=f"bilstm_{i+1}"
                )
            )
            if dropout_rate > 0:
                self.dropout_layers.append(
                    layers.Dropout(dropout_rate, name=f"dropout_{i+1}")
                )

        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    name=f"dense_{i+1}"
                )
            )

        # Output layer
        self.output_layer = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )

        _materialize_model(self, input_shape)
        logger.info(f"BidirectionalActionClassifier built with input shape {input_shape}")

    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs

        for i, bilstm in enumerate(self.bilstm_layers):
            x = bilstm(x)
            if training and i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)

        for dense in self.dense_layers:
            x = dense(x)

        return self.output_layer(x)


@keras.utils.register_keras_serializable(package="tensor_push_up")
class TemporalCNN(keras.Model):
    """
    1D CNN-based action classifier.

    Uses 1D convolutional layers to capture local temporal patterns
    followed by global pooling and dense layers.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        conv_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        dense_units: List[int] = [64],
        dropout_rate: float = 0.5,
        pooling: str = 'max',
        **kwargs
    ):
        """
        Initialize the Temporal CNN classifier.

        Args:
            input_shape: Input shape (window_size, features)
            num_classes: Number of output classes
            conv_filters: List of convolutional filters
            kernel_sizes: List of kernel sizes for each conv layer
            dense_units: List of dense layer units
            dropout_rate: Dropout rate
            pooling: Pooling type ('max' or 'avg')
        """
        kwargs.setdefault("name", "temporal_cnn")
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.pooling = pooling

        # Convolutional layers
        self.conv_layers = []
        self.pool_layers = []

        for i, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes)):
            self.conv_layers.append(
                layers.Conv1D(
                    filters,
                    kernel_size,
                    activation='relu',
                    padding='same',
                    name=f"conv1d_{i+1}"
                )
            )
            if pooling == 'max':
                self.pool_layers.append(
                    layers.MaxPooling1D(2, name=f"maxpool_{i+1}")
                )
            else:
                self.pool_layers.append(
                    layers.AveragePooling1D(2, name=f"avgpool_{i+1}")
                )

        # Global pooling
        self.global_pool = layers.GlobalMaxPooling1D(name="global_pool")

        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            self.dense_layers.append(
                layers.Dense(
                    units,
                    activation='relu',
                    name=f"dense_{i+1}"
                )
            )

        # Dropout
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate, name="dropout")

        # Output layer
        self.output_layer = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )

        _materialize_model(self, input_shape)
        logger.info(f"TemporalCNN built with input shape {input_shape}")

    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs

        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)

        x = self.global_pool(x)

        for dense in self.dense_layers:
            x = dense(x)

        if training and hasattr(self, 'dropout'):
            x = self.dropout(x)

        return self.output_layer(x)


@keras.utils.register_keras_serializable(package="tensor_push_up")
class TransformerBlock(layers.Layer):
    """
    Transformer encoder block for attention-based modeling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize Transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rate: Dropout rate
        """
        kwargs.setdefault("name", "transformer_block")
        super().__init__(**kwargs)

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            name="multi_head_attention"
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ], name="feed_forward")
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        """Forward pass."""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


@keras.utils.register_keras_serializable(package="tensor_push_up")
class TransformerClassifier(keras.Model):
    """
    Transformer-based action classifier.

    Uses self-attention mechanisms to model temporal dependencies.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_transformer_blocks: int = 2,
        mlp_units: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize the Transformer classifier.

        Args:
            input_shape: Input shape (window_size, features)
            num_classes: Number of output classes
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            mlp_units: List of MLP layer units
            dropout_rate: Dropout rate
        """
        kwargs.setdefault("name", "transformer_classifier")
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate

        # Input projection
        self.input_projection = layers.Dense(
            num_heads * 8,
            activation='relu',
            name='input_projection'
        )

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(num_heads * 8, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]

        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")

        # MLP layers
        self.mlp_layers = []
        for i, units in enumerate(mlp_units):
            self.mlp_layers.append(
                layers.Dense(
                    units,
                    activation='relu',
                    name=f"mlp_{i+1}"
                )
            )
            if dropout_rate > 0:
                self.mlp_layers.append(
                    layers.Dropout(dropout_rate, name=f"mlp_dropout_{i+1}")
                )

        # Output layer
        self.output_layer = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )

        _materialize_model(self, input_shape)
        logger.info(f"TransformerClassifier built with input shape {input_shape}")

    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.input_projection(inputs)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)

        x = self.global_pool(x)

        for layer in self.mlp_layers:
            x = layer(x)

        return self.output_layer(x)


# Model factory function
def create_model(
    model_type: str = 'lstm',
    input_shape: Tuple[int, int] = (30, 50),
    num_classes: int = 3,
    **kwargs
) -> keras.Model:
    """
    Factory function to create action classification models.

    Args:
        model_type: Type of model ('lstm', 'bilstm', 'cnn', 'transformer')
        input_shape: Input shape (window_size, features)
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters

    Returns:
        Compiled model
    """
    model_creators = {
        'lstm': ActionClassifier,
        'lstm_mlp': ActionClassifier,
        'bilstm': BidirectionalActionClassifier,
        'cnn': TemporalCNN,
        'transformer': TransformerClassifier
    }

    # Accept both 'lstm' and 'lstm' for compatibility
    valid_model_types = ['lstm', 'lstm_mlp', 'bilstm', 'cnn', 'transformer']
    if model_type not in valid_model_types:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {valid_model_types}")

    model_class = model_creators[model_type]
    model = model_class(
        input_shape=input_shape,
        num_classes=num_classes,
        **kwargs
    )

    logger.info(f"Created {model_type} model")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
    loss: str = 'sparse_categorical_crossentropy',
    metrics: Optional[List[str]] = None
) -> keras.Model:
    """
    Compile the model with specified optimizer and loss.

    Args:
        model: Model to compile
        learning_rate: Learning rate
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        loss: Loss function
        metrics: List of metric names

    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall']

    # Create optimizer
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Model compiled with optimizer={optimizer}, lr={learning_rate}, loss={loss}")
    return model


def create_callbacks(
    checkpoint_dir: str = "models/checkpoints",
    tensorboard_log_dir: str = "logs/tensorboard",
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5,
    checkpoint_monitor: str = "val_loss",
    use_tensorboard: bool = True
) -> List[callbacks.Callback]:
    """
    Create training callbacks.

    Args:
        checkpoint_dir: Directory to save checkpoints
        tensorboard_log_dir: TensorBoard log directory
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        checkpoint_monitor: Metric to monitor for checkpointing
        use_tensorboard: Whether to enable TensorBoard logging if available

    Returns:
        List of callbacks
    """
    from .utils import ensure_dir

    ensure_dir(checkpoint_dir)
    ensure_dir(tensorboard_log_dir)

    callback_list = []

    # Model checkpoint
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"{checkpoint_dir}/best_model.keras",
        monitor=checkpoint_monitor,
        save_best_only=True,
        mode='min' if 'loss' in checkpoint_monitor else 'max',
        verbose=1
    )
    callback_list.append(checkpoint_callback)

    # Early stopping
    early_stopping_callback = callbacks.EarlyStopping(
        monitor=checkpoint_monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stopping_callback)

    # Learning rate reduction
    reduce_lr_callback = callbacks.ReduceLROnPlateau(
        monitor=checkpoint_monitor,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callback_list.append(reduce_lr_callback)

    if use_tensorboard:
        if importlib.util.find_spec("tensorboard") is not None:
            tensorboard_callback = callbacks.TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=1,
                update_freq='epoch'
            )
            callback_list.append(tensorboard_callback)
            logger.info("TensorBoard callback enabled")
        else:
            logger.warning(
                "TensorBoard is not installed in the current environment. "
                "Skipping TensorBoard callback."
            )

    logger.info("Created callbacks: checkpoint, early_stopping, reduce_lr%s",
                ", tensorboard" if use_tensorboard and importlib.util.find_spec("tensorboard") is not None else "")
    return callback_list


def load_model_from_checkpoint(
    checkpoint_path: str,
    custom_objects: Optional[Dict] = None
) -> keras.Model:
    """
    Load model from checkpoint with security validation.

    Args:
        checkpoint_path: Path to model checkpoint
        custom_objects: Custom objects for model loading

    Returns:
        Loaded model
    """
    # Import security validation functions
    from .utils import validate_model_file

    # Validate model file
    if not validate_model_file(checkpoint_path):
        raise ValueError(f"Invalid model file: {checkpoint_path}")

    default_custom_objects = {
        'ActionClassifier': ActionClassifier,
        'BidirectionalActionClassifier': BidirectionalActionClassifier,
        'TemporalCNN': TemporalCNN,
        'TransformerBlock': TransformerBlock,
        'TransformerClassifier': TransformerClassifier
    }

    if custom_objects is None:
        custom_objects = {}
    custom_objects = {**default_custom_objects, **custom_objects}

    try:
        model = keras.models.load_model(
            checkpoint_path,
            custom_objects=custom_objects
        )

        logger.info(f"Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {str(e)}")
        raise ValueError(f"Invalid model file: {checkpoint_path}") from e


def export_model(
    model: keras.Model,
    export_dir: str,
    model_name: str = "action_classifier",
    formats: List[str] = ['saved_model']
):
    """
    Export model in various formats.

    Args:
        model: Model to export
        export_dir: Export directory
        model_name: Name for the exported model
        formats: List of export formats ('saved_model', 'h5', 'onnx')
    """
    from .utils import ensure_dir
    ensure_dir(export_dir)

    for format_name in formats:
        if format_name == 'saved_model':
            saved_model_path = f"{export_dir}/{model_name}"
            if hasattr(model, "export"):
                model.export(saved_model_path)
            else:
                model.save(saved_model_path)
            logger.info(f"Exported as SavedModel to {saved_model_path}")

        elif format_name == 'h5':
            model.save(f"{export_dir}/{model_name}.h5")
            logger.info(f"Exported as H5 to {export_dir}/{model_name}.h5")

        elif format_name == 'onnx':
            try:
                import tf2onnx
                onnx_model, _ = tf2onnx.convert.from_keras(
                    model,
                    output_path=f"{export_dir}/{model_name}.onnx"
                )
                logger.info(f"Exported as ONNX to {export_dir}/{model_name}.onnx")
            except ImportError:
                logger.warning("tf2onnx not installed, skipping ONNX export")
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")

        elif format_name == 'tflite':
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()

                with open(f"{export_dir}/{model_name}.tflite", 'wb') as f:
                    f.write(tflite_model)

                logger.info(f"Exported as TFLite to {export_dir}/{model_name}.tflite")
            except Exception as e:
                logger.error(f"TFLite export failed: {e}")


# Action class labels
ACTION_CLASSES = {
    0: 'pushup',
    1: 'jumping_jack',
    2: 'other'
}

ACTION_LABELS = {
    'pushup': 0,
    'jumping_jack': 1,
    'other': 2
}


def get_action_class_name(class_id: int) -> str:
    """Get action class name from ID."""
    return ACTION_CLASSES.get(class_id, 'unknown')


def get_action_class_id(class_name: str) -> int:
    """Get action class ID from name."""
    return ACTION_LABELS.get(class_name.lower(), 2)  # Default to 'other'


class ModelInference:
    """
    Wrapper for model inference with post-processing.

    Handles model loading, prediction, and result smoothing.
    """

    def __init__(
        self,
        model_path: str,
        window_size: int = 30,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize model inference wrapper.

        Args:
            model_path: Path to model checkpoint
            window_size: Input sequence window size
            smoothing_window: Window size for prediction smoothing
            confidence_threshold: Minimum confidence for predictions
        """
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        # Load model
        self.model = load_model_from_checkpoint(model_path)
        self.input_shape = self.model.input_shape[1:]  # (window_size, features)

        # Prediction buffer for smoothing
        self.prediction_buffer = []

        logger.info(f"ModelInference initialized with window_size={window_size}")

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict action class from features.

        Args:
            features: Feature array of shape (window_size, num_features)

        Returns:
            Tuple of (action_class, confidence)
        """
        # Ensure correct shape
        if features.shape[0] != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {features.shape[0]}")

        # Add batch dimension
        x = np.expand_dims(features, axis=0)

        # Predict
        predictions = self.model.predict(x, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = float(predictions[0][class_id])

        # Smooth predictions
        self.prediction_buffer.append(class_id)
        if len(self.prediction_buffer) > self.smoothing_window:
            self.prediction_buffer.pop(0)

        # Get smoothed prediction
        if len(self.prediction_buffer) >= self.smoothing_window:
            smoothed_class = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
        else:
            smoothed_class = class_id

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            smoothed_class = get_action_class_id('other')

        action_name = get_action_class_name(smoothed_class)

        return action_name, confidence

    def reset(self):
        """Reset prediction buffer."""
        self.prediction_buffer = []

    def get_model(self) -> keras.Model:
        """Get underlying model."""
        return self.model
