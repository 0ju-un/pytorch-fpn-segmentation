import argparse
import random
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

from tqdm import tqdm

from dataset import MyDataset
from FPN import FPN
from loss import *

## Parser 생성
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=25, type=int, dest="batch_size")
parser.add_argument("--num_workers", default=0, type=int, dest="num_workers")
parser.add_argument("--n_class", default=3, type=int, dest="n_class")

parser.add_argument("--root_dir", default="", type=str, dest="root_dir")

parser.add_argument("--mode", default="FPN", type=str, dest="mode")
parser.add_argument("--model_path", default="./ckpt_dir_/best_model.pth", type=str, dest="model_path")


args = parser.parse_args()

## 트레이닝 파라메터 설정
n_class = 3

# lr = args.lr
batch_size = args.batch_size
# num_epoch = args.num_epoch
num_workers = args.num_workers

root_dir = args.root_dir
data_dir = os.path.join(root_dir, 'dataset')
ckpt_dir = os.path.join(root_dir, 'test_ckpt_dir')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mode = args.mode
backbone = "resnext50"
model_path = args.model_path

base_threshold = .0

seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

global_start = time.time()

## 데이터셋 생성
trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

test_set = MyDataset(os.path.join(data_dir, "test"), transform=trans)
dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

## 네트워크 생성
if mode=="FPN":
    model = FPN(encoder_name=backbone,
                decoder_pyramid_channels=256,
                decoder_segmentation_channels=128,
                classes=n_class,
                dropout=0.3,
                activation='sigmoid',
                final_upsampling=4,
                decoder_merge_policy='add')## Optimizer 설정

## 네트워크 불러오기
model.to(device)
model.eval()
state = torch.load(model_path)

model.load_state_dict(state['state_dict'])

meter = Meter(ckpt_dir, base_threshold)

start = time.time()
for batch in tqdm(dataloader):
    images, targets = batch

    images = images.to(device)
    outputs = model(images)

    outputs = outputs.detach().cpu()
    meter.update("val", targets, outputs)

dices, iou = meter.get_metrics("val")
print("***** Prediction done in {} sec.; IoU: {}, Dice: {} ***** \n(total elapsed time: {} sec.) ". \
      format(int(time.time() - start), iou, dices[0]["dice_all"], int(time.time() - global_start)))