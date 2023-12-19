import argparse
import random
from PIL import Image, ImageOps, ImageEnhance
import glob
import torch
import logging
import os
import numpy as np
from resnet import  wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from torch.nn import functional as F
import torchvision.transforms as transforms

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss

def loss_fucntion_last(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    item = 0
    loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                    b[item].view(b[item].shape[0], -1)))
    return loss

def loss_fucntion_bn(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a[item] = torch.amax(a[item], dim=(2, 3))
        b[item] = torch.amax(b[item], dim=(2, 3))
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss

class PACSDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        data = np.load(root)
        
        self.transform = transform
        # load dataset
        self.img_paths = data["train_set_path"]
        # for item in in_domain_type:
        #     for selected_class in os.listdir(f'{config["PACS_root"]}/train/{item}/'):
        #         if selected_class == normal_class :
        #             continue
        #         self.img_paths += glob.glob(f'{config["PACS_root"]}/train/{item}/{selected_class}/*.jpg')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path= self.img_paths[idx]
        img = Image.open(config["PACS_root"] + img_path).convert('RGB')
        img = self.transform(img)

        return img, 0

IMAGE_SIZE = 256
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.Resampling.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.Transform.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.Resampling.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.Transform.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.Resampling.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.Transform.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.Resampling.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.Transform.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.Resampling.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def augpacs(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
  aug_list = [
      autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness
  ]
  severity = random.randint(0, severity)

  ws = np.float32(np.random.dirichlet([1] * width))
  m = np.float32(np.random.beta(alpha, alpha))
  preprocess_img = preprocess(image)
  mix = torch.zeros_like(preprocess_img)
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess_img + m * mix
  return mixed

class AugMixDatasetPACS(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess):
    self.dataset = dataset
    self.preprocess = preprocess
    self.gray_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
        transforms.Grayscale(3)
    ])
  def __getitem__(self, i):
    x, _ = self.dataset[i]
    return self.preprocess(x), augpacs(x, self.preprocess), self.gray_preprocess(x)

  def __len__(self):
    return len(self.dataset)
  

def train(normal_class, anomaly_class, running_times = 0):
    logging.info(normal_class)
    batch_size = 16
    image_size = 256

    labels_dict = config["PACS_class_to_idx"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(device)


    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])

    # train_path = f'{config["PACS_root"]}/train/photo/' +normal_class
    
    train_path = f'../domain-generalization-for-anomaly-detection/data/unsupervised/20231218-PACS-{normal_class}-{anomaly_class}.npz'
    train_data = PACSDataset(root=train_path, transform=resize_transform)
    train_data = AugMixDatasetPACS(train_data, preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

    if os.path.exists(f'./checkpoints/many-versus-many/test{running_times}') == False:
       os.mkdir(f'./checkpoints/many-versus-many/test{running_times}')


    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for normal, augmix_img, gray_img in train_dataloader:
            normal = normal.to(device)  # (3,256,256)
            inputs_normal = encoder(normal) # [(256,64,64), (512,32,32), (1024,16,16)]
            bn_normal = bn(inputs_normal) # (2048,8,8)
            outputs_normal = decoder(bn_normal)  # [(256,64,64), (512,32,32), (1024,16,16)]


            augmix_img = augmix_img.to(device) # (3,256,256)
            inputs_augmix = encoder(augmix_img) # [(256,64,64), (512,32,32), (1024,16,16)]
            bn_augmix = bn(inputs_augmix) # (2048,8,8)
            outputs_augmix = decoder(bn_augmix) # [(256,64,64), (512,32,32), (1024,16,16)]

            gray_img = gray_img.to(device) # (3,256,256)
            inputs_gray = encoder(gray_img)
            bn_gray = bn(inputs_gray)
            outputs_gray = decoder(bn_gray)

            # 对应论文的 L_abs
            loss_bn = loss_fucntion([bn_normal], [bn_augmix]) + loss_fucntion([bn_normal], [bn_gray])

            # 对应论文的 L_lowf
            loss_last = loss_fucntion_last(outputs_normal, outputs_augmix) + loss_fucntion_last(outputs_normal, outputs_gray)
            # 对应论文的 L_ori
            loss_normal = loss_fucntion(inputs_normal, outputs_normal)
            loss = loss_normal*0.9 + loss_bn*0.05 + loss_last*0.05

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        logging.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
    
    ckp_path = f'./checkpoints/many-versus-many/test{running_times}/PACS_DINL_{normal_class}_{anomaly_class}_epochs={epochs}_lr={learning_rate}_cnt={running_times}.pth'
    torch.save({'bn': bn.state_dict(),
                'decoder': decoder.state_dict()}, ckp_path)

    return


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
    logging.getLogger().setLevel(logging.INFO)
    args = argparse.ArgumentParser()
    args.add_argument("--epochs",type=int,default=20)
    args.add_argument("--learning_rate",type=float,default=0.005)
    args.add_argument("--gpu",type=str,default="0")
    args.add_argument("--running_times",type=int,default=0)
    args = args.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train("0123", "456", args.running_times)
    # train("456", "0123", args.running_times)
    # train("0246", "135", args.running_times)
    # train("135", "0246", args.running_times)

    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp.add_function(augpacs)
    # lp_wrapper  = lp(train)
    # lp_wrapper("0123", "456", 0)
    # lp.print_stats()
    
