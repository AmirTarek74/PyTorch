import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN,self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained= True, aux_logits= False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        for name,parm in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bais" in name:
                parm.requires_grad = True
            else:
                parm.requires_grad = self.train_CNN
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)

    def forward(self,imgs):
        features = self.inception(imgs)
        features = features.logits  # Accessing the main output tensor
        return self.drop_out(self.relu(features))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.drop_out = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.drop_out(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        return self.linear(hiddens)
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encodercnn = EncoderCNN(embed_size)
        self.decoderrnn = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, imgs, captions):
        features = self.encodercnn(imgs)
        output = self.decoderrnn(features, captions)

        return output

    def caption_image(self, image, vocab, max_length=50):
        result = []
        image = image.unsequeeze(0)
        with torch.no_grad():
            x = self.encodercnn(image).unsequeeze(0)
            states = None

            for i in range(max_length):
                hidden, states = self.decoderrnn.lstm(x, states)
                output = self.decoderrnn.linear(hidden.unsequeeze(0))
                predicted = output.argmax(1)
                result.append(predicted.item())
                x = self.decoderrnn.embed(predicted).unsequeeze(0)
                if vocab.itos[predicted.item()]=="<EOS>":
                    break
        return [vocab.itos[word] for word in result]
