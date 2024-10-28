import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import save_checkpoint, load_checkpoint
from data import get_data
from model import CNNtoRNN

def train():
    
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    root_dir = "/kaggle/input/flickr8kimagescaptions/flickr8k/images"
    captions_file = "/kaggle/input/flickr8kimagescaptions/flickr8k/captions.txt"
    test_ratio = 0.2
    train_loader, test_loader, data = get_data(root_dir,captions_file,transform,test_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False
    train_CNN = False

    embed_size = 256
    hidden_size = 256
    vocab_size = len(data.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr= learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=data.vocab.stoi["<PAD>"])

    model.train()
    for epoch in range(num_epochs):
        

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
            }
            save_checkpoint(checkpoint)

        model.train()
        train_loss = 0
        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            train_loss += loss.item()
            
            

            opt.zero_grad()
            loss.backward(loss)
            opt.step()
        print(f"Train Loss for Epoch {epoch+1} = {train_loss/len(train_loader):.2f}")

        test_loss = 0
        for idx, (imgs, captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            test_loss += loss.item()
        
        print(f"Test Loss for Epoch {epoch+1} = {test_loss/len(test_loader):.2f}")
            
        

if __name__ == "__main__":
    train()