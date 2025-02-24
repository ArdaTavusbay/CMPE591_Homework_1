import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hw1MLPDataset(Dataset):
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # one hot = +4
        input_dim = 49152 + 4
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, img, action):
        batch_size = img.shape[0]
        img_flat = img.view(batch_size, -1)
        action = action.long()
        action_onehot = F.one_hot(action, num_classes=4).float()
        x = torch.cat([img_flat, action_onehot], dim=1)
        return self.model(x)

def train():
    # Hyperparameters
    epochs = 200
    batch_size = 32
    lr = 1e-4

    train_dataset = Hw1MLPDataset(split="training_set")
    val_dataset = Hw1MLPDataset(split="validation_set")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for imgs, actions, targets in train_loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            predictions = model(imgs, actions)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, actions, targets in val_loader:
                imgs = imgs.to(device)
                actions = actions.to(device)
                targets = targets.to(device).float()
                predictions = model(imgs, actions)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item() * imgs.size(0)
        avg_val_loss = total_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    torch.save(model.state_dict(), "hw1_1.pt")
    print("Model saved as hw1_1.pt")

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (MLP)")
    plt.legend()
    plt.savefig("loss_plot_hw1_1.png")
    plt.show()
    print("Loss plot saved as loss_plot_hw1_1.png")

def test():
    model = MLP().to(device)
    model.load_state_dict(torch.load("hw1_1.pt", map_location=device))
    model.eval()

    test_dataset = Hw1MLPDataset(split="test_set")
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
    # train()
    test()
