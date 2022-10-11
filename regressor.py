import torch
from tqdm import tqdm
import random
from util import *


def logistic_loss(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    logistic loss: Mean of -y*log(y_cap) - (1-y)*log(1 - y_cap)
    """
    return torch.mean(-label * torch.log(prediction) - (1 - label) * torch.log(1 - prediction))


def scale_train(x: torch.Tensor):
    x_max = torch.max(x, dim=0).values + 0.000000001
    x_min = torch.min(x, dim=0).values + 0.0000000001
    return scale_predict(x, x_max, x_min), x_max, x_min


def scale_predict(x: torch.Tensor, x_max: torch.Tensor, x_min: torch.Tensor):
    return (x - x_min)/(x_max - x_min)


class BinaryLogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # We force output to be one, since we are doing binary logistic regression
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Sigmoid()
        )
        # Initialize weights. Note that this is not strictly necessary,
        # but you should test different initializations per lecture

        def init_weights(linear):
            if type(linear) == torch.nn.Linear:
                torch.nn.init.uniform_(linear.weight)

        self.linear.apply(init_weights)

    def forward(self, features: torch.Tensor):
        # We predict a number by multipling by the coefficients
        # and then take the sigmoid to turn the score as logits
        return self.linear(features)

    def predict(self, features: torch.Tensor):
        with torch.no_grad():
            return self.forward(features) > 0.5


def training_loop_no_pad(
        num_epochs,
        batch_size,
        train_features,
        train_labels,
        dev_features,
        dev_labels,
        optimizer,
        model
):
    samples = list(zip(train_features, train_labels))
    random.shuffle(samples)
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    print("Training...")
    for i in range(num_epochs):
        losses = []
        for batch in tqdm(batches):
            # Empty the dynamic computation graph
            features, labels = zip(*batch)
            features = torch.stack(features)
            labels = torch.stack(labels)
            optimizer.zero_grad()
            # Run the model
            logits = model(features)
            # Compute loss
            loss = logistic_loss(torch.squeeze(logits), labels)
            # In this logistic regression example,
            # this entails computing a single gradient
            loss.backward()
            # Backpropogate the loss through our model

            # Update our coefficients in the direction of the gradient.
            optimizer.step()
            # For logging
            losses.append(loss.item())

        # Estimate the f1 score for the development set
        dev_f1 = f1_score(model.predict(dev_features), dev_labels)
        print(f"epoch {i}, loss: {sum(losses) / len(losses)}")
        print(f"Dev F1 {dev_f1}")

    # Return the trained model
    return model
