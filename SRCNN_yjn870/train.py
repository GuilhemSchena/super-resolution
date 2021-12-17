import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda")

    torch.manual_seed(args.seed)

    # On crée le modèle
    model = SRCNN().to(device)
    # MSE loss
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    # Datasets
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # Initialisation
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    ### Sauvegarde des résultas
    adress = '/content/drive/MyDrive/SYS843/Github/sr/SRCNN_yjn870/results/'
    name_file_result = 'result_x' + str(args.scale) + '_lr' + str(args.lr) + '_batch' + str(args.batch_size) + '_epoch' + str(args.num_epochs) + '.csv'

    # Itération par epoch
    for epoch in range(args.num_epochs):
        # Entraînement        
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data
                # Charger sur le GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                # SR
                preds = model(inputs)
                # Calcul du coût
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        # Evaluation
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        # Sauvegarde des données :
        path_svg_model = '/content/drive/MyDrive/SYS843/Github/sr/SRCNN_yjn870/svg_models/srcnn_x' + str(args.scale) + '_epoch' + str(epoch) + 'sur' + str(args.num_epochs) + '_lr' + str(args.lr) + '_batch' + str(args.batch_size)
        torch.save(model.state_dict(), path_svg_model)
        
        # Ecrire dans le csv
        with open(adress + name_file_result, 'a', encoding='UTF8', newline='') as f:
          writer = csv.writer(f, delimiter=';')
          writer.writerow([epoch, epoch_losses.avg, epoch_psnr.avg.item()])

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))


    
    


