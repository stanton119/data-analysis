import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol, Dict, List, Tuple, Any, Optional, runtime_checkable


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the interface for multi-task learning models."""

    task_names: List[str]
    task_types: Dict[str, str]

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_names: List[str],
        task_types: Dict[str, str],
        **kwargs,
    ) -> None:
        """
        Initialize a multi-task learning model.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions
            task_names (List[str]): List of task names
            task_types (Dict[str, str]): Dictionary mapping task names to types ('binary', 'regression')
            **kwargs: Additional model-specific parameters
        """
        ...

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to predictions
        """
        ...


def get_model(
    model_name: str,
    num_tabular_features: int,
    task_names: List[str],
    task_types: Dict[str, str],
    model_params: Dict[str, Any] = None,
) -> ModelProtocol:
    """
    Factory function to get a model instance based on its name.

    Args:
        model_name (str): Name of the model to instantiate
        num_tabular_features (int): Number of tabular features (common to all models)
        task_names (List[str]): List of task names for multi-task models
        task_types (Dict[str, str]): Dictionary mapping task names to types
        model_params (Dict[str, Any], optional): Dictionary of model-specific parameters
            Possible keys include:
            - hidden_dims (List[int]): List of hidden layer dimensions
            - num_experts (int): Number of experts for MoE models
            - num_gates (int): Number of gates for MMoE models
            - expert_dims (List[int]): Dimensions for expert networks
            - gate_dims (List[int]): Dimensions for gate networks
            - target_task (str): For SingleTaskModel, the specific task to train

    Returns:
        ModelProtocol: Instantiated model
    """
    # Initialize model_params if not provided
    if model_params is None:
        model_params = {}

    # Get common parameters with defaults
    hidden_dims = model_params.get("hidden_dims", [64, 32])

    # Model-specific instantiation
    if model_name == "SingleTaskModel":
        # Get the target task for the single task model
        target_task = model_params.get("target_task")
        if target_task is None:
            raise ValueError("target_task must be specified for SingleTaskModel")

        # Ensure the target task is in the task_names list
        if target_task not in task_names:
            raise ValueError(f"Target task '{target_task}' not found in task_names")

        return SingleTaskModel(
            input_dim=num_tabular_features,
            hidden_dims=hidden_dims,
            task_names=task_names,
            task_types=task_types,
            target_task=target_task,
        )
        
    elif model_name == "MultiSingleTaskModel":
        return MultiSingleTaskModel(
            input_dim=num_tabular_features,
            hidden_dims=hidden_dims,
            task_names=task_names,
            task_types=task_types,
        )

    elif model_name == "SharedBottomModel":
        return SharedBottomModel(
            input_dim=num_tabular_features,
            hidden_dims=hidden_dims,
            task_names=task_names,
            task_types=task_types,
        )

    elif model_name == "MixtureOfExperts":
        # Get model-specific parameters with defaults
        num_experts = model_params.get("num_experts", 4)

        return MixtureOfExperts(
            input_dim=num_tabular_features,
            hidden_dims=hidden_dims,
            task_names=task_names,
            task_types=task_types,
            num_experts=num_experts,
        )

    elif model_name == "MultiGateMixtureOfExperts":
        # Get model-specific parameters with defaults
        num_experts = model_params.get("num_experts", 4)
        expert_dims = model_params.get("expert_dims", hidden_dims)
        gate_dims = model_params.get("gate_dims", [64])

        return MultiGateMixtureOfExperts(
            input_dim=num_tabular_features,
            hidden_dims=hidden_dims,
            task_names=task_names,
            task_types=task_types,
            num_experts=num_experts,
            expert_dims=expert_dims,
            gate_dims=gate_dims,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")


class SingleTaskModel(nn.Module, ModelProtocol):
    """
    Single Task Model for multi-task learning.

    This model implements the "Single model per task" approach where a separate model
    is trained for each task independently. While it's part of the multi-task framework,
    it actually focuses on a single target task and ignores others.

    As described in the README, this approach:
    - Is simple to implement and maintain
    - Has no interference between tasks
    - Can use task-specific architectures optimized for each task
    - Is easy to debug and interpret

    However, it:
    - Provides no knowledge sharing between tasks
    - Requires more parameters overall
    - Cannot leverage correlations between related tasks
    - Has higher computational and memory requirements for deployment
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_names: List[str],
        task_types: Dict[str, str],
        target_task: str,
    ):
        """
        Initialize the Single Task Model.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions
            task_names (List[str]): List of all task names (only target_task will be used)
            task_types (Dict[str, str]): Dictionary mapping task names to types ('binary', 'regression')
            target_task (str): The specific task this model will focus on
        """
        super().__init__()
        # Store all task names for compatibility with the ModelProtocol
        self.task_names = task_names
        self.task_types = task_types
        self.target_task = target_task

        # Create model layers for the target task
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Add the final output layer
        layers.append(nn.Linear(prev_dim, 1))

        # If binary classification, add sigmoid activation
        if self.task_types[target_task] == "binary":
            self.add_sigmoid = True
        else:
            self.add_sigmoid = False

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to predictions
                                    (only contains the target task)
        """
        # Pass input through the model
        output = self.model(x)

        # Apply sigmoid for binary classification
        if self.add_sigmoid:
            output = torch.sigmoid(output)

        # Return a dictionary with only the target task
        return {self.target_task: output}


class SharedBottomModel(nn.Module, ModelProtocol):
    """
    Shared Bottom Model for multi-task learning.

    This model implements the "hard parameter sharing" approach where lower layers
    are shared across all tasks, and task-specific output layers are used for each task.

    As described in the README, this approach:
    - Is parameter efficient by reducing overall model size
    - Enables knowledge transfer between related tasks
    - Helps prevent overfitting through implicit regularization
    - Provides a single model deployment for multiple tasks
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_names: List[str],
        task_types: Dict[str, str],
    ):
        """
        Initialize the Shared Bottom Model.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions for the shared bottom
            task_names (List[str]): List of task names
            task_types (Dict[str, str]): Dictionary mapping task names to types ('binary', 'regression')
        """
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types

        # Create shared bottom layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared_bottom = nn.Sequential(*layers)

        # Create task-specific output layers
        self.task_heads = nn.ModuleDict()
        for task_name in task_names:
            self.task_heads[task_name] = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to predictions
        """
        # Pass input through shared bottom layers
        shared_features = self.shared_bottom(x)

        # Pass shared features through task-specific heads
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](shared_features)

            # Apply appropriate activation based on task type
            if self.task_types[task_name] == "binary":
                task_output = torch.sigmoid(task_output)

            outputs[task_name] = task_output

        return outputs


class Expert(nn.Module):
    """
    Expert network for the Mixture of Experts model.

    Each expert is a simple feed-forward neural network that processes the input
    and produces a feature representation.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initialize an expert network.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Expert output of shape (batch_size, output_dim)
        """
        return self.network(x)


class MixtureOfExperts(nn.Module, ModelProtocol):
    """
    Mixture of Experts (MoE) Model for multi-task learning.

    This model implements the "One-gate Mixture of Experts" approach where:
    - Multiple expert networks process the input in parallel
    - A single gating network determines the weight of each expert for the input
    - The weighted combination of expert outputs is used for task-specific predictions

    As described in the README, this approach:
    - Is more flexible than hard parameter sharing
    - Can handle tasks with different relationships
    - Allows experts to specialize in different input patterns
    - Is more parameter efficient than separate models
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_names: List[str],
        task_types: Dict[str, str],
        num_experts: int = 4,
    ):
        """
        Initialize the Mixture of Experts Model.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions for each expert
            task_names (List[str]): List of task names
            task_types (Dict[str, str]): Dictionary mapping task names to types ('binary', 'regression')
            num_experts (int, optional): Number of expert networks. Defaults to 4.
        """
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types
        self.num_experts = num_experts

        # Create expert networks
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dims) for _ in range(num_experts)]
        )

        # Get the output dimension of experts
        expert_output_dim = self.experts[0].output_dim

        # Create a single gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            # No softmax here as it will be applied in the forward pass
        )

        # Create task-specific output layers
        self.task_heads = nn.ModuleDict()
        for task_name in task_names:
            self.task_heads[task_name] = nn.Linear(expert_output_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to predictions
        """
        batch_size = x.size(0)

        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        # Stack expert outputs: (num_experts, batch_size, expert_output_dim)
        expert_outputs = torch.stack(expert_outputs)

        # Get gating weights and apply softmax: (batch_size, num_experts)
        gate_weights = F.softmax(self.gate(x), dim=1)

        # Reshape gate weights for broadcasting: (batch_size, num_experts, 1)
        gate_weights = gate_weights.unsqueeze(-1)

        # Permute expert outputs for proper broadcasting: (batch_size, num_experts, expert_output_dim)
        expert_outputs = expert_outputs.permute(1, 0, 2)

        # Weighted sum of expert outputs: (batch_size, expert_output_dim)
        combined_output = torch.sum(gate_weights * expert_outputs, dim=1)

        # Pass combined output through task-specific heads
        outputs = {}
        for task_name in self.task_names:
            task_output = self.task_heads[task_name](combined_output)

            # Apply appropriate activation based on task type
            if self.task_types[task_name] == "binary":
                task_output = torch.sigmoid(task_output)

            outputs[task_name] = task_output

        return outputs


class MultiGateMixtureOfExperts(nn.Module, ModelProtocol):
    """
    Multi-gate Mixture of Experts (MMoE) Model for multi-task learning.

    This model implements the approach from "Modeling Task Relationships in Multi-task Learning 
    with Multi-gate Mixture-of-Experts" (Ma et al., 2018) where:
    - Multiple expert networks process the input in parallel
    - Each task has its own gating network to determine the weight of each expert
    - The weighted combination of expert outputs is used for task-specific predictions

    As described in the README, this approach:
    - Provides task-specific routing for more flexibility
    - Better handles tasks with different relationships
    - Reduces negative transfer between unrelated tasks
    - Can model complex task relationships
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_names: List[str],
        task_types: Dict[str, str],
        num_experts: int = 4,
        expert_dims: List[int] = None,
        gate_dims: List[int] = None,
    ):
        """
        Initialize the Multi-gate Mixture of Experts Model.

        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions for each expert
            task_names (List[str]): List of task names
            task_types (Dict[str, str]): Dictionary mapping task names to types ('binary', 'regression')
            num_experts (int, optional): Number of expert networks. Defaults to 4.
            expert_dims (List[int], optional): Hidden dimensions for expert networks. Defaults to hidden_dims.
            gate_dims (List[int], optional): Hidden dimensions for gate networks. Defaults to [64].
        """
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types
        self.num_experts = num_experts
        
        # Use default dimensions if not provided
        if expert_dims is None:
            expert_dims = hidden_dims
        if gate_dims is None:
            gate_dims = [64]

        # Create expert networks
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_dims) for _ in range(num_experts)]
        )

        # Get the output dimension of experts
        expert_output_dim = self.experts[0].output_dim

        # Create task-specific gating networks
        self.gates = nn.ModuleDict()
        for task_name in task_names:
            layers = []
            prev_dim = input_dim
            
            for gate_dim in gate_dims:
                layers.append(nn.Linear(prev_dim, gate_dim))
                layers.append(nn.ReLU())
                prev_dim = gate_dim
                
            layers.append(nn.Linear(prev_dim, num_experts))
            # No softmax here as it will be applied in the forward pass
            self.gates[task_name] = nn.Sequential(*layers)

        # Create task-specific output layers
        self.task_heads = nn.ModuleDict()
        for task_name in task_names:
            self.task_heads[task_name] = nn.Linear(expert_output_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to predictions
        """
        batch_size = x.size(0)

        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        # Stack expert outputs: (num_experts, batch_size, expert_output_dim)
        expert_outputs = torch.stack(expert_outputs)
        
        # Permute expert outputs for proper broadcasting: (batch_size, num_experts, expert_output_dim)
        expert_outputs = expert_outputs.permute(1, 0, 2)

        # Process each task with its own gate
        outputs = {}
        for task_name in self.task_names:
            # Get task-specific gating weights: (batch_size, num_experts)
            gate_weights = F.softmax(self.gates[task_name](x), dim=1)
            
            # Reshape gate weights for broadcasting: (batch_size, num_experts, 1)
            gate_weights = gate_weights.unsqueeze(-1)
            
            # Weighted sum of expert outputs for this task: (batch_size, expert_output_dim)
            task_combined_output = torch.sum(gate_weights * expert_outputs, dim=1)
            
            # Pass through task-specific head
            task_output = self.task_heads[task_name](task_combined_output)
            
            # Apply appropriate activation based on task type
            if self.task_types[task_name] == "binary":
                task_output = torch.sigmoid(task_output)
                
            outputs[task_name] = task_output

        return outputs


class MultiSingleTaskModel(nn.Module, ModelProtocol):
    """
    Multiple Single Task Models combined into one class.
    
    This model creates a separate model for each task but wraps them
    in a single class that follows the ModelProtocol interface.
    
    As described in the README, this "Single model per task" approach:
    - Is simple to implement and maintain
    - Has no interference between tasks
    - Can use task-specific architectures optimized for each task
    - Is easy to debug and interpret
    
    However, it:
    - Provides no knowledge sharing between tasks
    - Requires more parameters overall
    - Cannot leverage correlations between related tasks
    - Has higher computational and memory requirements for deployment
    
    This implementation allows for consistent comparison with multi-task models
    while maintaining the independence of each task's model.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_names: List[str],
        task_types: Dict[str, str],
    ):
        """
        Initialize the Multi Single Task Model.
        
        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (List[int]): List of hidden layer dimensions for each task model
            task_names (List[str]): List of task names
            task_types (Dict[str, str]): Dictionary mapping task names to types ('binary', 'regression')
        """
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types
        
        # Create a separate model for each task
        self.task_models = nn.ModuleDict()
        for task_name in task_names:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            layers.append(nn.Linear(prev_dim, 1))
            self.task_models[task_name] = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task-specific models.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping task names to predictions
        """
        outputs = {}
        for task_name in self.task_names:
            output = self.task_models[task_name](x)
            
            # Apply sigmoid for binary classification tasks
            if self.task_types[task_name] == "binary":
                output = torch.sigmoid(output)
                
            outputs[task_name] = output
            
        return outputs
