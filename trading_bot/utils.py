import os
import math
import logging

import pandas as pd
import numpy as np

from tensorflow.keras import backend as K


# Formats Position
format_position = lambda price: ("-$" if price < 0 else "+$") + "{0:.2f}".format(
    abs(price)
)


# Formats Currency
format_currency = lambda price: "${0:.2f}".format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """Displays training results"""
    if val_position == initial_offset or val_position == 0.0:
        logging.info(
            "Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}".format(
                result[0], result[1], format_position(result[2]), result[3]
            )
        )
    else:
        logging.info(
            "Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})".format(
                result[0],
                result[1],
                format_position(result[2]),
                format_position(val_position),
                result[3],
            )
        )


def show_eval_result(model_name, profit, initial_offset):
    """Displays eval results"""
    if profit == initial_offset or profit == 0.0:
        logging.info("{}: USELESS\n".format(model_name))
    else:
        logging.info("{}: {}\n".format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file"""
    df = pd.read_csv(stock_file)
    return list(df["Adj Close"])


def switch_k_backend_device(use_gpu=True):
    """Configures TensorFlow backend device (GPU or CPU).

    Args:
        use_gpu: If True, enables GPU usage. If False, forces CPU usage.
    """
    import tensorflow as tf

    if K.backend() == "tensorflow":
        if use_gpu:
            # Add CUDA paths to PATH for Windows
            # Check for system CUDA installation
            cuda_path = os.environ.get("CUDA_PATH", "")
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, "bin")
                if os.path.exists(cuda_bin) and cuda_bin not in os.environ.get(
                    "PATH", ""
                ):
                    os.environ["PATH"] = (
                        cuda_bin + os.pathsep + os.environ.get("PATH", "")
                    )

            # Also check for conda CUDA library path
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix:
                cuda_bin_path = os.path.join(conda_prefix, "Library", "bin")
                if os.path.exists(
                    cuda_bin_path
                ) and cuda_bin_path not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = (
                        cuda_bin_path + os.pathsep + os.environ.get("PATH", "")
                    )

            # Enable GPU if available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                    # Disable XLA JIT by default to avoid libdevice errors
                    # XLA requires CUDA libdevice which may not be available in all setups
                    # Users can enable it manually if they have proper CUDA setup
                    tf.config.optimizer.set_jit(False)
                    
                    # Also disable XLA via environment variable as a fallback
                    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=false'
                    
                    logging.info(f"Using GPU: {[gpu.name for gpu in gpus]}")
                    logging.info("GPU optimizations enabled: memory growth (XLA JIT disabled to avoid libdevice errors)")
                except RuntimeError as e:
                    logging.warning(f"GPU configuration error: {e}")
                    logging.info("Falling back to CPU")
            else:
                # Try to provide helpful debugging info
                logging.warning("No GPU devices found. Checking CUDA availability...")
                logging.warning(
                    f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}"
                )
                logging.warning("If you have a GPU, you may need to:")
                logging.warning("1. Install CUDA toolkit separately, or")
                logging.warning(
                    "2. Use TensorFlow 2.10 (last version with good Windows GPU support), or"
                )
                logging.warning(
                    "3. Use WSL2 with Linux TensorFlow for better GPU support"
                )
                logging.info("Falling back to CPU")
        else:
            # Force CPU usage
            logging.debug("Forcing CPU usage")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
