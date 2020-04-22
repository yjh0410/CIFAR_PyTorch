import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision import transforms as tf
import numpy as np
import matplotlib.pyplot as plt
from data import *
import os
import random
from models.net import cifar_net
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='cifarnet',
                        help='define model')
    parser.add_argument('--max_iterations', type=int, default='20000',
                        help='max training iterations')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=60, 
                        help='max epoch')
    parser.add_argument('--n_cpu', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='learning rate for training model')
    parser.add_argument('--momentum', type=float,
                        default=0.5, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight_decay for SGD')
    parser.add_argument('--gamma', type=float,
                        default=0.1, help='gamma for SGD')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--n_worker', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--path_to_train', type=str, default='data/cifar-10/cifar_train/')
    
    parser.add_argument('--path_to_test', type=str, default='data/cifar-10/cifar_test/')

    parser.add_argument('--path_to_save', type=str, default='weights/')
    
    parser.add_argument('--test', action='store_true', default=False)

    parser.add_argument('--trained_model', type=str, default='weights/')

    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# setup_seed(22)

    
def main():
    args = parse_args()

    path_to_save = os.path.join(args.path_to_save + args.model_name)
    os.makedirs(path_to_save, exist_ok=True)
    
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # images transform
    transform_train = tf.Compose([
                        tf.RandomCrop(32, padding=4),
                        tf.RandomHorizontalFlip(),
                        tf.ToTensor(),
                        tf.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                    ])

    transform_test = tf.Compose([
                            tf.ToTensor(),
                            tf.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                        ])

    # load trainset
    trainset = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    print('total train data size : ', len(trainset))

    model = cifar_net()
    model.to(device)

    # test
    if args.test:
        model.load_state_dict(torch.load(args.trained_model, map_location=device))
        model.to(device)
        model.eval()
        print('Test ...' )
        acc_test = test(model, device, transform_test)
        print('Done !!')
        print('Test Acc: %.4f' % (acc_test))
        exit(0)

    # basic setup
    base_lr = args.learning_rate
    tmp_lr = base_lr
    lr_step = [30, 50]
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    epoch_size = len(trainset) // args.batch_size

    # build loss function
    loss_f = nn.CrossEntropyLoss()

   
    best_acc = 0.0
    print("-------------- start training ----------------")
    for epoch in range(args.max_epoch):
        if epoch in lr_step:
            tmp_lr = tmp_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = tmp_lr

        for iter_i, (images, targets) in enumerate(trainloader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            pred = model(images)

            loss = loss_f(pred, targets)
            
            loss.backward()
            optimizer.step()

            if iter_i % 10 == 0:
                _, predicted = pred.max(1)
                total = targets.size(0)
                correct = predicted.eq(targets).sum().item()
                current_acc = 100.*correct / total
                print('[Epoch: %d/%d][Iter: %d/%d][Acc: %.4f][lr:%.8f]' 
                        % (epoch, args.max_epoch, iter_i, epoch_size, current_acc, tmp_lr),
                        flush=True)

        # test
        acc_test = test(model, device, transform_test)
        model.train()
        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(model.state_dict(), os.path.join(path_to_save, args.model_name + '_'+ str(round(best_acc, 4)) + '.pth'))


def test(model, device, transform_test):
    model.eval()
    
    # load testset
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        batch_iterator = iter(testloader)
        for images, targets in batch_iterator:
            images, targets = images.to(device), targets.to(device)
            pred = model(images)
            # print(torch.softmax(pred, 1))
            # pred_class = torch.argmax(pred, 1)[0]
            # img = images[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # plt.imshow(img)
            # plt.title(CIFAR_CLASS[pred_class])
            # plt.show()
            _, predicted = pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc_test = 100.*correct / total
    
    return acc_test

if __name__ == "__main__":
    main()