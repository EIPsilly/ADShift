import argparse
import re
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
from DGAD_inference_PACS_ATTA import evaluation_ATTA
from torchvision.datasets import ImageFolder
from collections import Counter

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
    def __init__(self, img_paths, labels, transform):
        
        self.transform = transform
        # load dataset
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path= self.img_paths[idx]
        img = Image.open(config["PACS_root"] + img_path).convert('RGB')
        img = self.transform(img)

        return img, self.labels[idx]

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

def test(encoder, bn, decoder, device, normal_class, lamda, _class_):
    
    list_results_AUROC = []
    list_results_AUPRC = []
    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_photo_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, normal_class = normal_class, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_photo {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_photo {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_art_painting_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, normal_class = normal_class, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_art {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_art {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_cartoon_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, normal_class = normal_class, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_cartoon {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_cartoon {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_OOD_sketch_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, normal_class = normal_class, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_sketch {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_sketch {:.4f}'.format(auprc))
    print(list_results_AUROC)
    print(list_results_AUPRC)


    return list_results_AUROC, list_results_AUPRC

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

    if args.domain_cnt == 3:
        data_path = f'../domain-generalization-for-anomaly-detection/data/pacs/unsupervised/3domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'
    if args.domain_cnt == 1:
        data_path = f'../domain-generalization-for-anomaly-detection/data/pacs/unsupervised/1domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'

    data = np.load(data_path)
    train_data = PACSDataset(img_paths=data["train_set_path"], labels = data["train_labels"], transform=resize_transform)
    train_data = AugMixDatasetPACS(train_data, preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    logging.info(f'train_labels: {Counter(data["train_labels"])}')
    
    img_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    
    val_data = PACSDataset(img_paths=data["val_set_path"], labels = data["val_labels"], transform=img_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)
    logging.info(f'val_labels: {Counter(data["val_labels"])}')
    
    
    test_data_ID_photo = PACSDataset(data["test_photo"], data["test_photo_labels"], transform=img_transforms)
    test_data_ID_art_painting = PACSDataset(data["test_art_painting"], data["test_art_painting_labels"], transform=img_transforms)
    test_data_ID_cartoon = PACSDataset(data["test_cartoon"], data["test_cartoon_labels"], transform=img_transforms)
    test_data_OOD_sketch = PACSDataset(data["test_sketch"], data["test_sketch_labels"], transform=img_transforms)
    
    logging.info(f'test_photo_labels: {Counter(data["test_photo_labels"])}')
    logging.info(f'test_art_painting_labels: {Counter(data["test_art_painting_labels"])}')
    logging.info(f'test_cartoon_labels: {Counter(data["test_cartoon_labels"])}')
    logging.info(f'test_sketch_labels: {Counter(data["test_sketch_labels"])}')

    global data_ID_photo_loader, data_ID_art_painting_loader, data_ID_cartoon_loader, data_OOD_sketch_loader
    
    data_ID_photo_loader = torch.utils.data.DataLoader(test_data_ID_photo, batch_size = 1, shuffle = False)
    data_ID_art_painting_loader = torch.utils.data.DataLoader(test_data_ID_art_painting, batch_size = 1, shuffle = False)
    data_ID_cartoon_loader = torch.utils.data.DataLoader(test_data_ID_cartoon, batch_size = 1, shuffle = False)
    data_OOD_sketch_loader = torch.utils.data.DataLoader(test_data_OOD_sketch, batch_size = 1, shuffle = False)
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

    file_name = f'domain_cnt={args.domain_cnt},normal_class={args.normal_class},learning_rate={args.learning_rate},epochs={args.epochs},cnt={running_times}'
    print(file_name)

    # if os.path.exists(f'./checkpoints/many-versus-many/test{running_times}') == False:
    #     os.mkdir(f'./checkpoints/many-versus-many/test{running_times}')
    if os.path.exists(f'./0417results{args.results_save_path}') == False:
        os.mkdir(f'./0417results{args.results_save_path}')
    # file_name = f'PACS_DINL_{normal_class}_{anomaly_class}_epochs={epochs}_lr={learning_rate}_cnt={running_times}'
    # ckp_path = f'./checkpoints/many-versus-many/test{running_times}/{file_name}.pth'
    
    import resnet_TTA
    inference_encoder, _ = resnet_TTA.wide_resnet50_2()
    inference_encoder.to(device)

    _class_ = int(normal_class[0])
    val_AUROC_list = []
    val_AUPRC_list = []
    train_results_loss = []
    val_max_metric = {"AUROC":-1,
                      "AUPRC":-1,
                      "epochs": None}
    test_results_list = []
    lamda = 0.5

    weight_dict = torch.load(f'./experiment{args.results_save_path}/{file_name}.pt')
    inference_encoder.load_state_dict(weight_dict["encoder"])
    bn.load_state_dict(weight_dict["bn"])
    decoder.load_state_dict(weight_dict["decoder"])
    inference_encoder.eval()
    test_AUROC, test_AUPRC = test(inference_encoder, bn, decoder, device, normal_class, lamda, _class_)
    test_metric = {}
    for idx, key in enumerate(["photo", "art_painting", "cartoon", "sketch"]):
        test_metric[key] = {
            "AUROC": test_AUROC[idx],
            "AUPRC": test_AUPRC[idx]
        }
    
    test_results_list.append(test_metric)

    print(val_AUROC_list)
    print(val_AUPRC_list)
    
    np.savez(f'./0417results{args.results_save_path}/{file_name}.npz',
             val_AUROC_list = np.array(val_AUROC_list),
             val_AUPRC_list = np.array(val_AUPRC_list),
             train_results_loss = np.array(train_results_loss),
             val_max_metric = np.array(val_max_metric),
             test_results_list = np.array(test_results_list),
             args = np.array(args.__dict__),)
    return


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
    logging.getLogger().setLevel(logging.INFO)
    args = argparse.ArgumentParser()
    args.add_argument("--epochs",type=int,default=2)
    args.add_argument("--learning_rate",type=float,default=0.005)
    args.add_argument("--gpu",type=str,default="0")
    args.add_argument("--running_times",type=int,default=0)
    args.add_argument("--results_save_path",type=str,default="/3domain_results")
    args.add_argument("--domain_cnt",type=int,default=3)
    args.add_argument("--normal_class", nargs="+", type=int, default=[2])
    args = args.parse_args()
    # args = args.parse_args(["--epochs", "2", "--results_save_path", "/3domain", "--gpu", "3"])
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    for file_name in os.listdir("experiment/3domain_results"):
    #    file_name = "domain_cnt=3,normal_class=[0],learning_rate=0.01,epochs=10,cnt=0.pt"
        # re.split(",", ) 
        for item in file_name.replace(".pt", "").split(","):
            key, value = item.split("=")
            if key == "learning_rate":
                args.learning_rate = float(value)
            elif key == "epochs":
                args.epochs = int(value)
            elif key == "cnt":
                args.cnt = int(value)
            elif key == "normal_class":
                args.normal_class = [int(value[1])]
    
        normal_class = str(args.normal_class[0])
        anomaly_class = "".join(map(str,(set([0,1,2,3,4,5,6]) - set(args.normal_class))))
        print("normal_class", normal_class)
        print("anomaly_class", anomaly_class)
    
        train(normal_class, anomaly_class, args.running_times)
    
    # train("0123", "456", args.running_times)
    # train("456", "0123", args.running_times)
    # train("0246", "135", args.running_times)
    # train("135", "0246", args.running_times)

    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp.add_function(augpacs)
    # lp_wrapper  = lp(train)
    # lp_wrapper("0123", "456", 0)
    # lp.print_stats()
    
