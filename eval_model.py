from torch.utils.data import DataLoader
import tqdm
from model import Net
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from utils import get_optimizer


if __name__ == '__main__':
    device = torch.device('cpu')
    train_data_path = test_data_path = './data'
    test_data = MNIST(test_data_path, False, transforms.ToTensor(), download=False)
    model = Net()
    optimizer = get_optimizer(model)

    for task_id in range(10):
        checkpoint_path = "checkpoints/task-%03d.pth" % task_id
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        batch_size = checkpoint['batch_size']

        model.eval()

        dataloader = tqdm.tqdm(DataLoader(test_data, batch_size, True),
                               desc='Eval (task {})'.format(task_id),
                               ncols=80, leave=True)
        correct = 0
        acc_list = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(1)
            correct += pred.eq(y).sum().item()
        accuracy = 100. * correct / (len(dataloader) * batch_size)
        acc_list.append(accuracy)
        print(accuracy)
    print(acc_list)

