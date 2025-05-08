import torch
import torch.nn as nn
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NeuralNet(nn.Module):
    """Simple neural network for chatbot intent classification."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_rate: float = 0.2):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Size of input layer.
            hidden_size (int): Size of hidden layers.
            num_classes (int): Number of output classes.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        
        Raises:
            ValueError: If input parameters are invalid.
        """
        super(NeuralNet, self).__init__()

        # Validate inputs
        if input_size <= 0 or hidden_size <= 0 or num_classes <= 0:
            logging.error("Input, hidden, or output sizes must be positive")
            raise ValueError("Input, hidden, and output sizes must be positive")
        if not 0 <= dropout_rate <= 1:
            logging.error("Dropout rate must be between 0 and 1")
            raise ValueError("Dropout rate must be between 0 and 1")

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)
        
        logging.info(
            f"Initialized NeuralNet: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_classes={num_classes}, "
            f"dropout_rate={dropout_rate}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out
