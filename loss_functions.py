import torch
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LossFunctions:
    """
    Custom loss functions for explainable deep anomaly detection.

    This class provides an implementation of custom loss functions as described in the research paper
    'Explainable Deep Anomaly Detection with Sequential Hypothesis Testing for Robotic Sewer
    Inspection'. It includes the velocity-threshold and Flow Theory methods for anomaly detection in
    sewer pipe inspection.

    ...

    Attributes
    ----------
    device : torch.device
        Device to use for tensor operations (CPU or CUDA)
    velocity_threshold : float
        Velocity threshold value for anomaly detection
    flow_theory_alpha : float
        Flow theory alpha value
    flow_theory_beta : float
        Flow theory beta value

    Methods
    -------
    velocity_threshold_loss(self, y_pred, y_true, velocity, flow)
        Velocity threshold loss function
    flow_theory_loss(self, y_pred, y_true, velocity, flow)
        Flow theory loss function
    bce_loss(self, y_pred, y_true)
        Binary cross-entropy loss function
    ...

    """

    def __init__(self, velocity_threshold: float = 0.5, flow_theory_alpha: float = 0.2, flow_theory_beta: float = 0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.velocity_threshold = velocity_threshold
        self.flow_theory_alpha = flow_theory_alpha
        self.flow_theory_beta = flow_theory_beta

    def velocity_threshold_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, velocity: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Velocity threshold loss function.

        This function implements the velocity threshold method for anomaly detection. It compares the predicted
        anomaly score (y_pred) with the ground truth (y_true) and calculates the loss using the velocity and
        flow information.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted anomaly scores (probabilities)
        y_true : torch.Tensor
            Ground truth anomaly labels (0 for normal, 1 for anomaly)
        velocity : torch.Tensor
            Velocity values
        flow : torch.Tensor
            Flow values

        Returns
        -------
        torch.Tensor
            Velocity threshold loss value

        """
        # Calculate the velocity difference
        vel_diff = velocity[..., 1:] - velocity[..., :-1]

        # Apply the velocity threshold
        is_anomaly = vel_diff > self.velocity_threshold

        # Calculate the flow rate
        flow_rate = (flow[..., 1:] - flow[..., :-1]) / vel_diff

        # Combine the velocity and flow conditions
        anomaly_condition = is_anomaly | (flow_rate > 0)

        # Calculate the binary cross-entropy loss
        loss = self.bce_loss(y_pred, y_true)

        # Apply the anomaly condition
        loss = loss * anomaly_condition.float()

        return loss.mean()

    def flow_theory_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, velocity: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Flow theory loss function.

        This function implements the flow theory method for anomaly detection. It uses the predicted anomaly
        scores (y_pred) and ground truth (y_true), along with the velocity and flow information, to calculate
        the loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted anomaly scores (probabilities)
        y_true : torch.Tensor
            Ground truth anomaly labels (0 for normal, 1 for anomaly)
        velocity : torch.Tensor
            Velocity values
        flow : torch.Tensor
            Flow values

        Returns
        -------
        torch.Tensor
            Flow theory loss value

        """
        # Calculate the velocity difference
        vel_diff = velocity[..., 1:] - velocity[..., :-1]

        # Calculate the flow rate
        flow_rate = (flow[..., 1:] - flow[..., :-1]) / vel_diff

        # Apply the flow theory formula
        anomaly_score = self.flow_theory_alpha * torch.exp(-self.flow_theory_beta * flow_rate)

        # Calculate the binary cross-entropy loss with the anomaly score
        loss = self.bce_loss(anomaly_score, y_true)

        return loss.mean()

    def bce_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy loss function.

        This function calculates the binary cross-entropy loss between the predicted values (y_pred) and the
        ground truth (y_true).

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values (probabilities)
        y_true : torch.Tensor
            Ground truth labels

        Returns
        -------
        torch.Tensor
            Binary cross-entropy loss value

        """
        return torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    def weighted_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted loss function.

        This function calculates the weighted loss between the predicted values (y_pred) and the ground truth
        (y_true) using the provided weights.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values (probabilities)
        y_true : torch.Tensor
            Ground truth labels
        weights : torch.Tensor
            Loss weights for each data point

        Returns
        -------
        torch.Tensor
            Weighted loss value

        """
        # Calculate the binary cross-entropy loss
        loss = self.bce_loss(y_pred, y_true)

        # Apply the weights
        loss = loss * weights

        return loss.mean()

# Example usage
if __name__ == "__main__":
    # Create instance of LossFunctions
    loss_funcs = LossFunctions()

    # Example data
    y_pred = torch.rand(100, 1)
    y_true = torch.randint(0, 2, (100, 1))
    velocity = torch.rand(100, 10)
    flow = torch.rand(100, 10)
    weights = torch.rand(100)

    # Velocity threshold loss
    velocity_loss = loss_funcs.velocity_threshold_loss(y_pred, y_true, velocity, flow)
    logger.info(f"Velocity Threshold Loss: {velocity_loss.item():.4f}")

    # Flow theory loss
    flow_loss = loss_funcs.flow_theory_loss(y_pred, y_true, velocity, flow)
    logger.info(f"Flow Theory Loss: {flow_loss.item():.4f}")

    # Binary cross-entropy loss
    bce_loss = loss_funcs.bce_loss(y_pred, y_true)
    logger.info(f"Binary Cross-Entropy Loss: {bce_loss.item():.4f}")

    # Weighted loss
    weighted_loss = loss_funcs.weighted_loss(y_pred, y_true, weights)
    logger.info(f"Weighted Loss: {weighted_loss.item():.4f}")