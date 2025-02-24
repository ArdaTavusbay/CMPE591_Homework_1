import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReconstructionDataset(Dataset):
    def __init__(self, split="training_set"):
        data_dir = os.path.join(os.getcwd(), split)
        self.positions = torch.load(os.path.join(data_dir, "positions.pt"))
        self.actions = torch.load(os.path.join(data_dir, "actions.pt"))
        self.imgs_before = torch.load(os.path.join(data_dir, "imgs_before.pt"))
        self.imgs_after = torch.load(os.path.join(data_dir, "imgs_after.pt"))
        self.imgs_before = self.imgs_before.float() / 255.0
        self.imgs_after = self.imgs_after.float() / 255.0

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        pos = self.positions[idx]
        norm_x = pos[0] / 1.2
        norm_y = (pos[1] + 0.6) / 1.2
        pos_norm = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        return self.imgs_before[idx], self.actions[idx], pos_norm, self.imgs_after[idx]

class AutoEncoder(nn.Module):
    def __init__(self, num_actions=4):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(4096 + 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, action, pos):
        x = self.encoder(image)
        act_onehot = F.one_hot(action.long(), num_classes=4).float()
        cond = torch.cat([pos, act_onehot], dim=1)
        x = torch.cat([x, cond], dim=1)
        x = self.fc(x)
        x = self.decoder(x)
        return x

def train():
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.0025

    train_ds = ReconstructionDataset("training_set")
    val_ds = ReconstructionDataset("validation_set")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img_b, act, pos, img_a in train_loader:
            img_b = img_b.to(device)
            act = act.to(device)
            pos = pos.to(device)
            img_a = img_a.to(device)
            optimizer.zero_grad()
            pred = model(img_b, act, pos)
            loss = criterion(pred, img_a)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * img_b.size(0)
        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for img_b, act, pos, img_a in val_loader:
                img_b = img_b.to(device)
                act = act.to(device)
                pos = pos.to(device)
                img_a = img_a.to(device)
                pred = model(img_b, act, pos)
                loss = criterion(pred, img_a)
                running_val_loss += loss.item() * img_b.size(0)
        val_loss = running_val_loss / len(val_ds)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "hw1_3.pt")
            print(f"Saved best model at epoch {epoch+1}")

    plt.figure(figsize=(8,4))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("reconstruction_loss.png")
    plt.show()

def test(num_samples=5):
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load("hw1_3.pt", map_location=device))
    model.eval()

    test_ds = ReconstructionDataset("test_set")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    criterion = nn.MSELoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for img_b, act, pos, img_a in test_loader:
            img_b = img_b.to(device)
            act = act.to(device)
            pos = pos.to(device)
            img_a = img_a.to(device)
            pred = model(img_b, act, pos)
            loss = criterion(pred, img_a)
            total_loss += loss.item()
            count += 1
    avg_test_loss = total_loss / count
    print("Final Test Loss: {:.6f}".format(avg_test_loss))

    samples = []
    for img_b, act, pos, img_a in test_loader:
        samples.append((img_b, act, pos, img_a))
        if len(samples) >= num_samples:
            break

    fig, axs = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    fig.suptitle("Reconstruction Results", fontsize=16)

    for i, (img_b, act, pos, img_a) in enumerate(samples):
        img_b = img_b.to(device)
        act = act.to(device)
        pos = pos.to(device)
        img_a = img_a.to(device)
        with torch.no_grad():
            pred = model(img_b, act, pos)
        img_b_np = img_b.squeeze(0).cpu().numpy().transpose(1,2,0)
        img_a_np = img_a.squeeze(0).cpu().numpy().transpose(1,2,0)
        pred_np = pred.squeeze(0).cpu().numpy().transpose(1,2,0)
        axs[0, i].imshow(img_b_np)
        axs[0, i].set_title("Initial (Before)")
        axs[0, i].axis("off")
        axs[1, i].imshow(img_a_np)
        axs[1, i].set_title("Ground Truth (After)")
        axs[1, i].axis("off")
        axs[2, i].imshow(pred_np)
        axs[2, i].set_title("Predicted (After)")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_results.png")
    plt.show()
    print("Saved reconstruction visualization as reconstruction_results.png")

if __name__ == "__main__":
    #train()
    test()