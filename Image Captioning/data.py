import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms as transforms


spacy_eng = spacy.load("en_core_web_sm")

# 1- Define Vocabulary mapping each word to an index
class Vocabulary:
    def __init__(self, freq_th):        # freq_th to pay attention how many times the word repated
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_th = freq_th

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


    def build_vocabulary(self,captions):
        freqs = {}
        idx = 4 
        for caption in captions:
            for word in self.tokenizer_eng(caption):
                freqs[word] = freqs.get(word,0) + 1
       
                if freqs[word]==self.freq_th:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[tok] if tok in self.stoi  else self.stoi["<UNK>"] for tok in tokenized_text
        ]

# 2- Define Dataset calss
class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_th=5):
        super().__init__()
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.transform = transform
        
        self.vocab = Vocabulary(freq_th)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_path)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])


        return img,torch.tensor(numericalized_caption)
    
# 3- add padding to each sequence to make them having the same length which is equalt to the largest length in the batch 
class collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions,batch_first=False,padding_value=self.pad_idx)

        return imgs, captions
    
def get_data(
        root_dir,
        captions_file,
        transform,
        test_ratio=0.2,
        batch_size=32,
        shuffle=True
):
    data = FlickerDataset(root_dir,captions_file,transform)

    
    test_size = int(len(data)* test_ratio)
    train_size = len(data)- test_size
    
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    pad_idx = data.vocab.stoi["<PAD>"]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True,collate_fn=collate(pad_idx))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,drop_last=True,collate_fn=collate(pad_idx))



    return train_loader, test_loader, data





