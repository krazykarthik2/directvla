import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import VLA_Transformer

def train():
    print("Loading Dobot VLA demonstration dataset...")
    obs = np.load('data/obs.npy').astype(np.float32)
    intent = np.load('data/intent.npy').astype(np.float32)
    proprio = np.load('data/proprio.npy').astype(np.float32)
    actions = np.load('data/actions.npy').astype(np.float32)

    print(f"Dataset summary: {obs.shape[0]} samples.")
    
    dataset = TensorDataset(
        torch.from_numpy(obs),
        torch.from_numpy(intent),
        torch.from_numpy(proprio),
        torch.from_numpy(actions)
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLA_Transformer().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 15
    print(f"Training VLA policy on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for b_obs, b_intent, b_proprio, b_action in dataloader:
            b_obs = b_obs.to(device)
            b_intent = b_intent.to(device)
            b_proprio = b_proprio.to(device)
            b_action = b_action.to(device)
            
            optimizer.zero_grad()
            preds = model(b_obs, b_intent, b_proprio)
            loss = criterion(preds, b_action)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), "vla_model.pth")
    print("Training finished! Saved weights to vla_model.pth")

if __name__ == "__main__":
    train()
