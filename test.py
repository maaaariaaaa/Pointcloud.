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

ROOT_PATH = os.path.abspath('./')


test_transform = transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])


def test(model, test_loader, device, is1=True):
        model.eval()
        #EVALUATE
        total_correct = 0
        total_testset = 0
        valid_loss = 0
        with torch.no_grad():
            for i,data in enumerate(test_loader, 0):
                points, target = data
                #target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)
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
        

def __main__():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = PointNetCls(k=4)
    model = Pointnet2(4, normal_channel=False)
    model = model.float()
    model.load_state_dict(torch.load(os.path.join(ROOT_PATH, 'data', 'checkpoint_cutout_pointnet2.pt'), map_location=device))
    model.to(device)    
    test_ds = Dataset("test", transform=test_transform)
    batchSize=32
    test_loader = DataLoader(dataset=test_ds, batch_size=batchSize, shuffle=True)
    #test(model, test_loader, device, is1=True)
    test(model, test_loader, device, is1=False)
    

if __name__ == '__main__':
    __main__()