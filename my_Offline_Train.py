import os
import torch
import torch.nn as nn
import torch.nn.functional as F # a bunch of functions, e.g. convolution
import pandas as pd
import numpy as np
from collections import defaultdict # https://docs.python.org/3/library/collections.html#collections.defaultdict
from torch.func import functional_call
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# Load dataset of residuals
csv_path = os.path.join(os.path.dirname(__file__), 'residuals.csv')
df = pd.read_csv(csv_path)

input_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()
residuals = df[['residual']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(input_data, residuals, test_size=0.5)


# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# MAML Training Loop

def main():
    device = torch.device("cpu")
    
    # Hyperparameters
    learning_rate = 1e-3
    input_dim = 4
    output_dim = 1
    hidden_dim = 128
    num_layers = 4

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    for param in residual_mlp.parameters():
        param.requires_grad = False
    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=learning_rate) # lr = learning rate, the optimizer
    residual_criterion = nn.MSELoss()
    
    X_batch = torch.tensor(X_train, dtype=torch.float32)
    y_target = torch.tensor(y_train, dtype=torch.float32)

    for p in residual_mlp.parameters(): p.requires_grad = True
    for _ in range(1000):
        residual_optimizer.zero_grad() # optimizer object
        prediction = residual_mlp(X_batch) # gives data to network to make a prediction
        loss = residual_criterion(prediction, y_target)
        loss.backward() # calculates gradient
        residual_optimizer.step() # one optimization step to update parameters
    for p in residual_mlp.parameters(): p.requires_grad = False
    torch.save(residual_mlp.state_dict(), "heating_model_2.pth")

    test_data = torch.tensor(X_test, dtype=torch.float32)
    residual_mlp.eval()
    with torch.no_grad():
        outputs = residual_mlp(test_data)
        target_predicted = np.array(outputs.squeeze().tolist())
    mse = mean_squared_error(target_predicted, y_test)
    print(y_test.size)
    len_test = len(y_test)
    null_prediction = np.zeros((len_test,))
    print("MSE", mse)
    mse_null_hypothesis = mean_squared_error(y_test, null_prediction)
    print("MSE null hypothesis", mse_null_hypothesis)
    #for var_name in residual_optimizer.state_dict():
    #    print(var_name, '\t', residual_optimizer.state_dict()[var_name])
    """
    for epoch in range(epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        n_tasks_used = 0

        task_ids = np.random.choice(list(task_data.keys()), meta_batch_size, replace=False)

        for tid in task_ids:
            x, y = task_data[tid]

            permutation = torch.randperm(x.size(0))
            x, y = x[permutation], y[permutation]

            x_support, y_support = x[:K].to(device), y[:K].to(device)
            x_query, y_query = x[K:K+K].to(device), y[K:K+K].to(device)

            adapted_params = {
                name: param.clone() for name, param in model.named_parameters()
                }
            
            for _ in range(inner_steps):
                support_pred = functional_call(model, adapted_params, (x_support,)) # replacing parameters in model with provided ones
                loss = F.mse_loss(support_pred, y_support)
                grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True) 
                adapted_params = {
                    name: param - inner_lr * grad
                    for (name, param), grad in zip(adapted_params.items(), grads)
                }

            query_pred = functional_call(model, adapted_params, (x_query, ))
            task_loss = F.mse_loss(query_pred, y_query)
            meta_loss += task_loss
            n_tasks_used += 1
        
        if n_tasks_used == 0:
            continue
        
        meta_loss = meta_loss / n_tasks_used
        meta_loss.backward() # calcualtes gradient
        meta_optimizer.step()
        train_losses.append(meta_loss.item())

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print("Epoch {0} Meta Loss: {1}".format(epoch+1, meta_loss.item())) 

    # Save model
    save_dir = os.path.dirname(__file__)
    save_path = os.path.join(save_dir, "maml_cartpole_meta_init_{0}_{1}.pth".format(num_layers, hidden_dim))
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim, 
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
    }, save_path)

    print("Model saved")

    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Meta Loss (MSE)")
    plt.title("MAML Meta-Training Loss (CartPole Residuals)")
    plt.grid(True)
    plt.yscale("log")
    plt.show()
    """
if __name__ == "__main__":
    main()
    

