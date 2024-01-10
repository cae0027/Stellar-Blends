import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class DynamicCNN(nn.Module):
    def __init__(self, input_nodes, num_layers, nodes_per_layer):
        super(DynamicCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, nodes_per_layer[0], kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())

        for i in range(1, num_layers):
            self.layers.append(nn.Conv2d(nodes_per_layer[i - 1], nodes_per_layer[i], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())

        # Calculate the size of the flattened input based on the output size of the convolutional layers
        self.flatten_size = self._calculate_flatten_size(input_nodes)

        self.layers.append(nn.Flatten())

        # Adjust the sizes of the linear layers based on your requirements
        linear_layers = []
        for i in range(len(nodes_per_layer) - 1):
            linear_layers.append(nn.Linear(self.flatten_size, nodes_per_layer[i + 1]))
            linear_layers.append(nn.ReLU())
            self.flatten_size = nodes_per_layer[i + 1]

        self.layers.extend(linear_layers)
        self.layers.append(nn.Linear(self.flatten_size, 2))  # 2 classes for binary classification

    def _calculate_flatten_size(self, input_nodes):
        dummy_input = torch.randn(1, 1, input_nodes, input_nodes)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(dummy_output.size(0), -1).size(1)

    def _forward_conv(self, x):
        for layer in self.layers[:4 * num_layers]:  # Pass through convolutional layers only
            x = layer(x)
        return x

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = torch.tensor(self.df.iloc[idx, 0], dtype=torch.long)
        img_data = torch.tensor(self.df.iloc[idx, 1:].values, dtype=torch.float32).view(1, 10, 10)

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, label
# assign device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_and_evaluate_model(csv_path, input_nodes, num_layers, nodes_per_layer, num_epochs=100, batch_size=64, lr=0.001, test_size=0.2):
    dataset = CustomDataset(csv_path, transform=transforms.Normalize((0.5,), (0.5,)))
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = DynamicCNN(input_nodes=input_nodes, num_layers=num_layers, nodes_per_layer=nodes_per_layer)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data.detach().cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy for {csv_path}: {accuracy}")
    return accuracy

# Example usage
if __name__ == "__main__":
    # csv_file = '../../data/data-norm/max-only/raw_image_data.csv'  # the path to your CSV file
    # num_layers = 2
    # nodes_per_layer = [32,64]
    # train_and_evaluate_model(csv_file, input_nodes=10, num_layers=num_layers, nodes_per_layer=nodes_per_layer)

    # vary test sizes
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8]
    accuracy = {i:[] for i in test_sizes}
    for test_size in test_sizes:
        csv_file = '../../data/data-norm/max-only/norm_21.csv'  # the path to your CSV file
        num_layers = 2
        nodes_per_layer = [32,64]
        accur = train_and_evaluate_model(csv_file, input_nodes=10, num_layers=num_layers, nodes_per_layer=nodes_per_layer, test_size=test_size)
        accuracy[test_size].append(accur)
    
    # pickle save accuracy
    import pickle
    with open('../vary-test-size/CNN-accuracy.pkl', 'wb') as f:
        pickle.dump(accuracy, f, pickle.HIGHEST_PROTOCOL)

