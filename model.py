import torch
import torchaudio
from torch.utils.data import DataLoader
from torch import nn
from USDataset import US8KDataset

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
SAMPLE_RATE = 44100
NUM_SAMPLES = 44100
device = "cpu"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


def train(model, train_loader, optimiser, device, epochs):
    correctly_classified = 0
    total = 0
    for epochX in range(epochs):
        print(f"Epoch {epochX + 1}")
        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            optimiser.zero_grad()
            output = model(input)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correctly_classified += (predicted == target).sum().item()
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimiser.step()
            accuracy = 100.0 * correctly_classified / total
        print(' Accuracy in Epoch {}: {:.0f}% \n'.format(epochX, accuracy))
        print(f"loss: {loss.item()}")
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = US8KDataset(ANNOTATIONS_FILE,
                      AUDIO_DIR,
                      mel_spectrogram,
                      SAMPLE_RATE,
                      NUM_SAMPLES,
                      device)

    train_dataloader = DataLoader(usd, batch_size=BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNN().to(device)
    print(cnn)

    optimiser = torch.optim.AdamW(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "trainedCNN.pth")
    print("Trained feed forward net saved at trainedCNN.pth")


