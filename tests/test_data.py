import torchvision
import torch

def test_data():
    train_dataset = torchvision.datasets.MNIST(root=r'C:\Users\amali\Desktop\MLOps_kursus\Dag3\MNIST', train=True, download=False, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=r'C:\Users\amali\Desktop\MLOps_kursus\Dag3\MNIST', train=False, download=False, transform=torchvision.transforms.ToTensor())
    assert train_dataset.data.size()[0] == 60000 and test_dataset.data.size()[0] == 10000
    # assert that each datapoint has shape [1,28,28]
    assert train_dataset.data.size()[1:] == torch.Size([28,28]) and test_dataset.data.size()[1:] == torch.Size([28,28])
    # assert that all labels are represented
    assert all([(x in train_dataset.targets) for x in torch.tensor([0,1,2,3,4,5,6,7,8,9])]) and all([(x in test_dataset.targets) for x in torch.tensor([0,1,2,3,4,5,6,7,8,9])])


