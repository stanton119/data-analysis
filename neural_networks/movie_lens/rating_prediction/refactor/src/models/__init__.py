import importlib
import pytorch_lightning as pyl

MODEL_REGISTRY = {
    "NeuralColabFilterSigmoid": "models.neural_colab_filter_sigmoid",
    "NeuralColabFilter": "models.neural_colab_filter",
    "NeuralColabFilterNonLinear": "models.neural_colab_filter_non_linear",
    "NeuralInnerProduct": "models.neural_inner_product",
}


def get_model(architecture: str, **kwargs) -> pyl.LightningModule:
    """Dynamically loads the specified model."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {architecture}")

    module_path = MODEL_REGISTRY[architecture]
    module = importlib.import_module(module_path)
    model_class = getattr(module, "Model")  # Ensure each module has a 'Model' class
    return model_class(**kwargs)
