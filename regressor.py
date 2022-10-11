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


class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # We force output to be one, since we are doing binary logistic regression
        self.output_size = 1
        self.coefficients = torch.nn.Linear(input_dim, self.output_size)
        # Initialize weights. Note that this is not strictly necessary,
        # but you should test different initializations per lecture
        torch.nn.init.uniform_(self.coefficients.weight)
        self.coefficients.bias.data *= 0

    def forward(self, features: torch.Tensor):
        # We predict a number by multipling by the coefficients
        # and then take the sigmoid to turn the score as logits
        return torch.sigmoid(self.coefficients(features))

    def predict(self, features: torch.Tensor):
        return self.forward(features) > 0.5


def training_loop(
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
        dev_f1 = f1_score(predict(model, dev_features), dev_labels)
        print(f"epoch {i}, loss: {sum(losses) / len(losses)}")
        print(f"Dev F1 {dev_f1}")

    # Return the trained model
    return model

