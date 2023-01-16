import torch
from model_pointnet import PointNetCls, feature_transform_regularizer
from model_pointnet2 import Pointnet2
from dataloader import Dataset
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import *
import torch.nn.functional as F
import tqdm
from earlystopping import EarlyStopping

#Transformationen des Training-Datensatzes
train_transform = transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                Rotation(5),
                                Shear(5),
                                Normalize(),
                                ToTensor()
                              ])

#Transformationen des Validierungsdatensatzes
test_transform = transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

#Training des Models
def train(model, train_loader, val_loader=None,  epochs=300, batchSize=32, feature_transform=False, is1=True):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    num_batch = len(train_loader)
    for epoch in range(0, epochs):
        #Trainiere
        for i, data in enumerate(train_loader, 0):
            points, target = data
            #target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            model = model.train()
            if is1:
                pred, trans, trans_feat= model(points.float())      
            else:
                pred, _ =  model(points.float())       
            loss = F.nll_loss(pred, target)
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(batchSize)))

            if i % 10 == 0:
                with torch.no_grad():
                    j, data = next(enumerate(val_loader, 0))
                    points, target = data
                    #target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.to(device), target.to(device)
                    model = model.eval()
                    if is1:
                        pred, _, _= model(points.float())      
                    else:
                        pred, _ =  model(points.float())
                    loss = F.nll_loss(pred, target)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'test', loss.item(), correct.item()/float(batchSize)))

        torch.save(model.state_dict(), 'model_w.pt')

        #evaluiere
        total_correct = 0
        total_testset = 0
        valid_loss = 0
        with torch.no_grad():
            for i,data in enumerate(val_loader, 0):
                points, target = data
                #target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)
                model = model.eval()
                if is1:
                    pred, _, _ = model(points.float())      
                else:
                    pred, _ =  model(points.float())
                valid_loss += F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]
        
        print("val accuracy {}".format(total_correct / float(total_testset)))
        #Earlystopping
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

#Trainiert das Model. Je nach dem was auskommentiert mit Pointnet oder Pointnet++
def __main__():
    #model = PointNetCls(k=4)
    model = Pointnet2(4, normal_channel=False)
    model = model.float()
    train_ds = Dataset("train", transform=train_transform)
    val_ds = Dataset("val", transform=test_transform)
    batchSize=32
    train_loader = DataLoader(dataset=train_ds, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(dataset=val_ds, batch_size=batchSize, shuffle=True)
    #train(model, train_loader, test_loader, batchSize=batchSize, is1=True)
    train(model, train_loader, test_loader, batchSize=batchSize, is1=False)
    

if __name__ == '__main__':
    __main__()
