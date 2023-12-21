import argparse
import re
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import torch
import os
from torchvision.datasets import ImageFolder
from resnet_TTA import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
import torchvision.transforms as transforms
from test import evaluation_ATTA
from os import listdir
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
import yaml

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

labels_dict = config["PACS_idx_to_class"]

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def evaluation_ATTA(encoder, bn, decoder, dataloader,device, type_of_test, img_size, lamda=0.5, dataset_name='mnist', _class_=None):
    bn.eval()
    decoder.eval()
    gt_list_sp = []
    pr_list_sp = []

    if dataset_name == 'mnist':
        link_to_normal_sample = f'{config["mnist_grey_root"]}/training/' + str(_class_) #update the link here
        filenames = [f for f in listdir(link_to_normal_sample)]
        filenames.sort()
        link_to_normal_sample = f'{config["mnist_grey_root"]}/training/' + str(_class_) + '/' + filenames[0] #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")


    if dataset_name == 'mvtec':
        link_to_normal_sample = f'{config["mvtec_root"]}' + _class_ + '/train/good/000.png' #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name == 'PACS':
        link_to_normal_sample = f'{config["PACS_root"]}/train/photo/' + labels_dict[_class_] #update the link here
        filenames = [f for f in listdir(link_to_normal_sample)]
        filenames.sort()
        link_to_normal_sample = f'{config["PACS_root"]}/train/photo/' + labels_dict[_class_] + '/' + filenames[0] #update the link here
        normal_image = Image.open(link_to_normal_sample).convert("RGB")

    if dataset_name != 'mnist':
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)
        ])
    else:
        trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    normal_image = trans(normal_image)
    normal_image = torch.unsqueeze(normal_image, 0)



    with torch.no_grad():
        for sample in dataloader:
            img, label = sample[0], sample[1]

            if dataset_name != 'mvtec' and dataset_name != 'mvtec_ood':
                if int(label) in normal_class:
                    label = 0
                else:
                    label = 1
            else:
                label = int(torch.sum(label) != 0)


            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)

            normal_image = normal_image.to(device)
            img = img.to(device)
            inputs = encoder(img, normal_image, type_of_test, lamda=lamda)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.append(int(label))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        precision, recall, threshold = precision_recall_curve(gt_list_sp, pr_list_sp)
        auprc = auc(recall, precision)
    return auroc_sp, auprc

def test_PACS(_class_, model_name, running_times = 0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    name_dataset = labels_dict[_class_]
    print('Class: ', name_dataset)

    #load data
    size = 256
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    img_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])

    test_path_ID_photo = f'{config["PACS_root"]}/test/photo/' #update here
    test_path_ID_art_painting = f'{config["PACS_root"]}/test/art_painting/' #update here
    test_path_ID_cartoon = f'{config["PACS_root"]}/test/cartoon/' #update here
    test_path_OOD_sketch = f'{config["PACS_root"]}/test/sketch/' #update here

    test_data_ID_photo = ImageFolder(root=test_path_ID_photo, transform=img_transforms)
    test_data_ID_art_painting = ImageFolder(root=test_path_ID_art_painting, transform=img_transforms)
    test_data_ID_cartoon = ImageFolder(root=test_path_ID_cartoon, transform=img_transforms)
    test_data_OOD_sketch = ImageFolder(root=test_path_OOD_sketch, transform=img_transforms)

    data_ID_photo_loader = torch.utils.data.DataLoader(test_data_ID_photo, batch_size=1, shuffle=False)
    data_ID_art_painting_loader = torch.utils.data.DataLoader(test_data_ID_art_painting, batch_size=1, shuffle=False)
    data_ID_cartoon_loader = torch.utils.data.DataLoader(test_data_ID_cartoon, batch_size=1, shuffle=False)
    data_OOD_sketch_loader = torch.utils.data.DataLoader(test_data_OOD_sketch, batch_size=1, shuffle=False)

    # ckp_path_decoder = './checkpoints/' + 'PACS_DINL_' + name_dataset + '_19.pth'
    ckp_path_decoder = f'./checkpoints/many-versus-many/test{running_times}/{model_name}'

    #load model
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    #load checkpoint
    ckp = torch.load(ckp_path_decoder)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.load_state_dict(ckp['bn'], strict=False)
    decoder.eval()
    bn.eval()

    lamda = 0.5

    list_results_AUROC = []
    list_results_AUPRC = []
    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_photo_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_photo {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_photo {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_art_painting_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_art {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_art {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_cartoon_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_cartoon {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_cartoon {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_OOD_sketch_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample AUROC_sketch {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_sketch {:.4f}'.format(auprc))
    print(list_results_AUROC)
    print(list_results_AUPRC)


    return list_results_AUROC, list_results_AUPRC

# test_PACS(1)

args = argparse.ArgumentParser()
args.add_argument("--cnt", default=0, type=int)
args.add_argument("--gpu", default="0", type=str)
args = args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
running_times = args.cnt

normal_class = None
anomaly_class = None

normal_class = [0,1,2,3]
anomaly_class = [4,5,6]

AUROC_results = []
AUPRC_results = []
para_results = []
for model_name in sorted(os.listdir(f'checkpoints/many-versus-many/test{running_times}')):
    if not ("0123_456" in model_name):
        continue
    splits = re.split("_|\.pth", model_name)
    # normal_class = list(map(int, list(splits[2])))
    # anomaly_class = list(map(int, list(splits[3])))
    para = []
    for item in splits:
        if "=" in item:
            para.append(item.split("=")[1])

    auroc, auprc = test_PACS(normal_class[0], model_name, running_times)
    para_results.append(para)
    AUROC_results.append(auroc)
    AUPRC_results.append(auprc)
    print('===============================================')
    print('')
    print('')


np.savez(f"results/many-versus-many-results-{running_times}.npz", AUROC_results = np.array(AUROC_results), AUPRC_results = np.array(AUPRC_results), para_results = np.array(para_results))

# nohup python DGAD_inference_PACS_ATTA.py --cnt 4 --gpu 0 > DGAD_inference4.log 2>&1 &