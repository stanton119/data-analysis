import importlib
import pytorch_lightning as pyl

MODEL_REGISTRY = {
    "NeuralColabFilterSigmoid": "recsys.models.neural_colab_filter_sigmoid",
    "NeuralColabFilter": "recsys.models.neural_colab_filter",
    "NeuralColabFilterNonLinear": "recsys.models.neural_colab_filter_non_linear",
    "NeuralInnerProduct": "recsys.models.neural_inner_product",
    "NeuMF": "recsys.models.neumf",
    "MultVAE": "recsys.models.mult_vae",
    "DCN": "recsys.models.dcn",
    "LightGCN": "recsys.models.lightgcn",
    "WideAndDeep": "recsys.models.wide_and_deep",
    "AutoInt": "recsys.models.autoint",
}


def get_model(architecture: str, **kwargs) -> pyl.LightningModule:
    """Dynamically loads the specified model."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {architecture}")

    module_path = MODEL_REGISTRY[architecture]
    module = importlib.import_module(module_path)
    model_class = getattr(module, "Model")  # Ensure each module has a 'Model' class
    return model_class(**kwargs)
