import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def generate_tactile_data(n_samples=1000):
    data = []
    for _ in range(n_samples):
        # 1. sample tactile state parameters
        F_n = np.random.uniform(0.1, 10.0) # normal force (N)
        A = np.random.uniform(0.5, 5.0) # contact area (mm^2)
        delta = np.random.uniform(0.1, 2.0) # displacement (mm)
        x_c = np.random.uniform(8, 56) # contact center x (pixel)
        y_c = np.random.uniform(8, 56)

        # 2. generate tactile image (2x64x64)
        xx, yy = np.meshgrid(np.arange(64), np.arange(64))
        dist = np.sqrt((xx - x_c)**2 + (yy - y_c)**2)
        sigma = np.sqrt(A) * 2  # spread of the contact area
        disp_x = delta * np.exp(-dist**2 / (2*sigma**2)) * np.random.normal(1, 0.1)
        disp_y = delta * np.exp(-dist**2 / (2*sigma**2)) * np.random.normal(1, 0.1)
        tactile_img = np.stack([disp_x, disp_y], axis=0).astype(np.float32)

        # 3. generate pressure matrix (16x16)
        xx_p, yy_p = np.meshgrid(np.linspace(0, 63, 16), np.linspace(0, 63, 16))
        dist_p = np.sqrt((xx_p - x_c)**2 + (yy_p - y_c)**2)
        sigma_p = sigma / 4
        pressure = F_n * np.exp(-dist_p**2 / (2*sigma_p**2))
        pressure += np.random.normal(0, 0.1 * F_n, pressure.shape)
        pressure = np.maximum(pressure, 0).astype(np.float32)

        data.append({
        'tactile_img': tactile_img, # [2, 64, 64]

        'pressure_mat': pressure, # [16, 16]

        'state': np.array([F_n, A, delta, x_c, y_c], dtype=np.float32)
        })

    return data

class TactileDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tactile_img = torch.tensor(sample['tactile_img'], dtype=torch.float32)
        pressure_mat = torch.tensor(sample['pressure_mat'], dtype=torch.float32)
        state = torch.tensor(sample['state'], dtype=torch.float32)
        return tactile_img, pressure_mat, state

class PressPerceptionNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.tactile_conv = nn.Sequential(
            # 2x64x64 -> 16x64x64
            nn.Conv2d(2, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            # 16x64x64 -> 16x32x32
            nn.MaxPool2d(2), # 16x32x32

            # 16x32x32 -> 32x32x32
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            # 32x32x32 -> 32x16x16
            nn.MaxPool2d(2), # 32x16x16

            # 32x16x16 -> 64x16x16
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # 64x16x16 -> 64x1x1
            nn.AdaptiveAvgPool2d(1),
        )

        self.press_conv = nn.Sequential(
            # 1x16x16 -> 16x16x16
            nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            # 16x16x16 -> 32x16x16
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fusion = nn.Sequential(
            # tactile: 64, pressure: 32
            nn.Linear(64 + 32, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),


            nn.Linear(64, 5) # output: [F_n, A, delta, x_c, y_c]
        )

    def forward(self, tactile_img, pressure_mat):
        # tactile_img: [batch, 2, 64, 64]
        # pressure_mat: [batch, 1, 16, 16]

        tactile_feat = self.tactile_conv(tactile_img) # [batch, 64, 1, 1]
        # flatten
        tactile_feat = tactile_feat.view(tactile_feat.size(0), -1) # [batch, 64]

        pressure_feat = self.press_conv(pressure_mat) # [batch, 32, 1, 1]
        # flatten
        pressure_feat = pressure_feat.view(pressure_feat.size(0), -1) # [batch, 32]
        # fusion

        return self.fusion(torch.cat([tactile_feat, pressure_feat], dim=1))


if __name__ == "__main__":
    # test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PressPerceptionNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=50, factor=0.8
    )

    num_epochs = 2000
    batch_size = 64

    data = generate_tactile_data(n_samples=5120)

    train_dataset = TactileDataSet(data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        )

    model.train()


    for epoch in range(num_epochs):
        total_loss = 0.0

        for tactile_img, pressure_mat, state in train_loader:
            tactile_img = tactile_img.to(device)
            pressure_mat = pressure_mat.unsqueeze(1).to(device) # add channel dimension
            state = state.to(device)

            output = model(tactile_img, pressure_mat)
            loss = criterion(output, state)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * tactile_img.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
        scheduler.step(epoch_loss)

    # evaluate the model
    model.eval()
    correct = 0
    total = 0
    threshold = 0.05
    scale = torch.tensor([10.0, 5.0, 2.0, 64.0, 64.0], device=device)

    with torch.no_grad():
        for tactile_img, pressure_mat, state in train_loader:
            tactile_img = tactile_img.to(device)
            pressure_mat = pressure_mat.unsqueeze(1).to(device) # add channel dimension
            state = state.to(device)

            output = model(tactile_img, pressure_mat)
            # count total samples and correct predictions
            total += state.size(0)

            # normalize error
            error = (output - state).abs() / scale  # [B, 5]

            correct += (error < threshold).sum(dim=0)

    accuracy_per_dim  = correct / total
    overall_accuracy =  accuracy_per_dim.mean()
    print("Per-dim accuracy:", accuracy_per_dim.cpu().numpy())
    print("Overall accuracy:", overall_accuracy.item())


    plt.figure(figsize=(22, 9))

    with torch.no_grad():
        tactile_list = []
        pressure_list = []
        gt_list = []
        pred_list = []

        for i in range(5):
            tactile_img, pressure_mat, state = train_dataset[i]
            tactile_img = tactile_img.unsqueeze(0).to(device)
            pressure_mat = pressure_mat.unsqueeze(0).unsqueeze(0).to(device)
            state = state.to(device)

            output = model(tactile_img, pressure_mat)


            tactile_list.append(tactile_img[0].cpu().numpy())          # (2, 64, 64)
            pressure_list.append(pressure_mat.squeeze().cpu().numpy()) # 2d matrix (16, 16)

            # format to 2 decimal places for better visualization
            gt_vals = np.round(state.cpu().numpy(), 2)
            pred_vals = np.round(output.cpu().numpy()[0], 2) # remove batch dimension

            gt_list.append(gt_vals)
            pred_list.append(pred_vals)

        # first row: plot 5 tactile images
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            # add title with ground truth and inference values
            plt.title(f"Tactile {i}\nTarget: {gt_list[i]}\nInfer:  {pred_list[i]}", fontsize=10)
            plt.imshow(tactile_list[i][0], cmap='jet')
            plt.colorbar(orientation='horizontal', pad=0.15)

        # second row: plot 5 pressure matrices
        for i in range(5):
            plt.subplot(2, 5, i + 6)
            plt.title(f"Pressure {i}", fontsize=10)
            plt.imshow(pressure_list[i], cmap='jet')
            plt.colorbar(orientation='horizontal', pad=0.15)

    plt.tight_layout() # auto adjust subplot params for better layout
    plt.show()
