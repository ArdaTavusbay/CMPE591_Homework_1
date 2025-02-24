import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hw1CNNDataset(Dataset):
    def __init__(self, split="training_set"):
        data_dir = os.path.join(os.getcwd(), split)
        self.positions = torch.load(os.path.join(data_dir, "positions.pt"))
        self.actions = torch.load(os.path.join(data_dir, "actions.pt"))
        self.imgs = torch.load(os.path.join(data_dir, "imgs.pt"))
        self.imgs = self.imgs.float() / 255.0

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.actions[idx], self.positions[idx]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(8192 + 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, img, action):
        x = self.features(img)
        x = x.view(x.size(0), -1)
        action = action.long()
        action_onehot = F.one_hot(action, num_classes=4).float()
        x = torch.cat([x, action_onehot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def train():
    # Hyperparameters.
    epochs = 50
    batch_size = 16
    learning_rate = 1e-3

    train_dataset = Hw1CNNDataset(split="training_set")
    val_dataset = Hw1CNNDataset(split="validation_set")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, actions, targets in train_loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(imgs, actions)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, actions, targets in val_loader:
                imgs = imgs.to(device)
                actions = actions.to(device)
                targets = targets.to(device).float()
                outputs = model(imgs, actions)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        val_epoch_loss = val_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    torch.save(model.state_dict(), "hw1_2.pt")
    print("Model saved as hw1_2.pt")

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (CNN)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot_hw1_2.png")
    plt.show()
    print("Loss plot saved as loss_plot_hw1_2.png")

def test():
    model = CNN().to(device)
    model.load_state_dict(torch.load("hw1_2.pt", map_location=device))
    model.eval()

    test_dataset = Hw1CNNDataset(split="test_set")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for imgs, actions, target_pos in test_loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            target_pos = target_pos.to(device).float()
            pred_pos = model(imgs, actions)
            loss = criterion(pred_pos, target_pos)
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)
    
    avg_test_loss = total_loss / count
    print("Final Test Loss: {:.6f}".format(avg_test_loss))

    idx = torch.randint(len(test_dataset), (1,)).item()
    img, action, target_pos = test_dataset[idx]
    img = img.unsqueeze(0).to(device)
    action = torch.tensor([action]).to(device)
    with torch.no_grad():
        pred_pos = model(img, action)
    print(f"Test sample index: {idx}")
    print(f"Action (id): {action.item()}")
    print(f"Ground Truth Position: {target_pos.numpy()}")
    print(f"Predicted Position: {pred_pos.cpu().numpy()[0]}")

if __name__ == "__main__":
    #train()
    test()
