from datasets import load_dataset
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms

ds = load_dataset("zalando-datasets/fashion_mnist")
train = ds["train"]
test = ds["test"]

label2str = train.features["label"].int2str  # funzione per convertire da id a stringa

plt.figure(figsize=(6,4))

for i in range(6):
    sample = train[i]
    img = sample["image"]
    label = label2str(sample["label"])

    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()

class Fashion_MNIST(Dataset):
    def __init__(self, labels, imgs, transform = None, target_transform = None):
        self.labels = labels
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label

train_img = train["image"]
train_label = train["label"]

test_img = test["image"]
test_label = test["label"]

transform = transforms.ToTensor()

train_dataset = Fashion_MNIST(train_label, train_img, transform=transform)
test_dataset  = Fashion_MNIST(test_label,  test_img,  transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=32)



class ConvNet(LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


model = ConvNet(lr=1e-3)

trainer = Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto"
)

trainer.fit(model, train_dataloader, test_dataloader)