import numpy as np
import pandas as pd
import torch
import os
from torchvision.datasets import ImageFolder
from resnet_TTA import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
import torchvision.transforms as transforms
from test import evaluation_ATTA


with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

labels_dict = config["PACS_idx_to_class"]

def test_PACS(_class_, running_times = 0):
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

    test_path_ID = f'{config["PACS_root"]}/test/photo/' #update here
    test_path_OOD_art_painting = f'{config["PACS_root"]}/test/art_painting/' #update here
    test_path_OOD_cartoon = f'{config["PACS_root"]}/test/cartoon/' #update here
    test_path_OOD_sketch = f'{config["PACS_root"]}/test/sketch/' #update here

    test_data_ID = ImageFolder(root=test_path_ID, transform=img_transforms)
    test_data_OOD_art_painting = ImageFolder(root=test_path_OOD_art_painting, transform=img_transforms)
    test_data_OOD_cartoon = ImageFolder(root=test_path_OOD_cartoon, transform=img_transforms)
    test_data_OOD_sketch = ImageFolder(root=test_path_OOD_sketch, transform=img_transforms)

    data_ID_loader = torch.utils.data.DataLoader(test_data_ID, batch_size=1, shuffle=False)
    data_OOD_art_painting_loader = torch.utils.data.DataLoader(test_data_OOD_art_painting, batch_size=1, shuffle=False)
    data_OOD_cartoon_loader = torch.utils.data.DataLoader(test_data_OOD_cartoon, batch_size=1, shuffle=False)
    data_OOD_sketch_loader = torch.utils.data.DataLoader(test_data_OOD_sketch, batch_size=1, shuffle=False)

    ckp_path_decoder = f'checkpoints/one-versus-many/test{running_times}/PACS_DINL_{name_dataset}_19.pth'

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
    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_ID_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample Auroc_photo {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_photo {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_OOD_art_painting_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample Auroc_art {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_art {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_OOD_cartoon_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample Auroc_cartoon {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_cartoon {:.4f}'.format(auprc))

    auroc_sp, auprc = evaluation_ATTA(encoder, bn, decoder, data_OOD_sketch_loader, device,
                                               type_of_test='EFDM_test',
                                               img_size=256, lamda=lamda, dataset_name='PACS', _class_=_class_)
    list_results_AUROC.append(auroc_sp)
    list_results_AUPRC.append(auprc)
    print('Sample Auroc_sketch {:.4f}'.format(auroc_sp))
    print('Sample AUPRC_sketch {:.4f}'.format(auprc))
    print(list_results_AUROC)
    print(list_results_AUPRC)


    return list_results_AUROC, list_results_AUPRC

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# test_PACS(1)

AUROC_results = []
AUPRC_results = []
for running_times in range(10):
    one_auroc_results = []
    one_auprc_results = []
    for i in range(0,7):
        auroc, auprc = test_PACS(i, running_times)
        one_auroc_results.append(auroc)
        one_auprc_results.append(auprc)
        print('===============================================')
        print('')
        print('')
    AUROC_results.append(one_auroc_results)
    AUPRC_results.append(one_auprc_results)

np.savez("results/one-versus-many-results.npz", AUROC_results = np.array(AUROC_results), AUPRC_results = np.array(AUPRC_results))

# nohup python inference_PACS_ATTA.py > inference_PACS_ATTA.log 2>&1 &