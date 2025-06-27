from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import tokenizers
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import warnings

import ipdb


entities = ['abnormality', 'abscess', 'aerate', 'aorta', 'atelectasis', 'bronchiectasis', 'calcification', 'cardiomediastinal', \
            'cardiomegaly', 'catheter', 'chf', 'collapse', 'congestion', 'consolidation', 'contour', 'COPD', \
            'deformity', 'dilation', 'distention', 'edema', 'effusion', 'embolism', 'emphysema', 'engorgement', \
            'fibrosis', 'fracture', 'granuloma', 'hernia', 'hilar', 'hyperinflate', 'hemidiaphragm', 'infiltrate', \
            'mass','nodule', 'obscure', 'opacity', 'perihilar', 'pneumonia', 'pneumothorax', 'sarcoidosis', \
            'silhouette', 'thickening', 'tuberculosis', 'vasculature']
template1 = [219, 149, 152, 422, 158]       # there is no evidence of
template2 = [219, 149, 152]       # there is no


def pil_loader(path: str) -> Image.Image:
    """
    打开图像文件并转换为 RGB 格式，添加对损坏图像的容错处理。
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    #     return img.convert('RGB')
    try:
        # 打开文件，避免 ResourceWarning
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # 转换为 RGB 格式
    except Exception as e:
        # 捕获异常并发出警告
        warnings.warn(f"Skipping file {path} due to error: {e}")
        return None  # 返回 None，表示图像损坏或无法加载


class ContextBertDataset(Dataset):
    def __init__(
        self,
        data_root,
        max_caption_length: int = 256,
    ):
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.images_list, self.report_list, self.llm_out_list, self.attn_i_list, self.attn_j_list = self.read_csv()
        self.tokenizer = tokenizers.Tokenizer.from_file(os.path.join(self.data_root, "mimic_wordpiece.json"))
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.attn_weights_path = "../../../Data/MIMIC-CXR-weight-img"

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4721], std=[0.3037])
        ])

        # ipdb.set_trace()

    def __len__(self):
        return len(self.images_list)


    def _context_mask(self,tokens):
        masked_tokens = deepcopy(tokens)
        entity_pos = []
        mask_pos = []       # entity context mask position
        entity_exist = False

        for i in range(1, masked_tokens.shape[1]-1):
            if self.idxtoword[masked_tokens[0][i].item()] in entities:
                entity_exist = True
                break

        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break
            
            if masked_tokens[0][i-1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue
            
            if masked_tokens[0][i-1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue


            if self.idxtoword[masked_tokens[0][i].item()] in entities:
                entity_pos.append(i)
                
                for j in range(1, 3):
                    if i-j <= 0:
                        break
                    else:
                        if tokens[0][i-j] != 16:       # 16 is "."
                            if i-j not in mask_pos:
                                mask_pos.append(i-j)
                            if self.idxtoword[masked_tokens[0][i].item()] not in entities and masked_tokens[0][i-j] != 3:
                                masked_tokens[0][i-j] = 3

            prob = random.random()
            if not entity_exist:
                if prob < 0.75:
                    masked_tokens[0][i] = 3
            else:
                if prob < 0.7 and i not in entity_pos and i not in mask_pos:
                    masked_tokens[0][i] = 3
        
        for i in range(1, masked_tokens.shape[1]-1):                # mask entity on 75% prob
            if i in entity_pos:
                prob = random.random()
                if prob < 0.75:
                    masked_tokens[0][i] = 3

        return masked_tokens, mask_pos


    def __getitem__(self, index):
        # image = pil_loader(self.images_list[index])
        max_attempts = 4242  # 最大尝试次数，避免无限循环
        attempt = 0
        
        while attempt < max_attempts:
            image = pil_loader(self.images_list[index])
            if image is not None:
                break
            warnings.warn(f"Image at index {index} is invalid, trying next index.")
            index = (index + 1) % len(self.images_list)  # 循环到下一个索引
            attempt += 1
        
        if image is None:
            raise ValueError(f"Could not find valid image after {max_attempts} attempts.")
    
        image = self.transform(image)
        report = self.report_list[index].split('.')
        report_len = len(report)
        vicuna_output = self.llm_out_list[index]
        
        # 将 vicuna_output 转换为字符串，并处理空值
        if pd.isna(vicuna_output):  # 处理 NaN 或 None
            vicuna_output = ""
        else:
            vicuna_output = str(vicuna_output)  # 确保是字符串
        
        sent = ""
        add_prob = random.random()
        if add_prob < 0.8:
            location = random.randint(0, report_len)
            for i in range(0, location):
                sent += report[i]
                sent += "."
            sent += vicuna_output
            for i in range(location, report_len):
                sent += report[i]
                sent += "."
        else:
            sent = self.report_list[index]
        sent = sent.replace("..", ".")
        sent = '[CLS] ' + sent
        self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        self.tokenizer.enable_padding(length=self.max_caption_length)

        encoded = self.tokenizer.encode(sent)
        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)
        self.weights = torch.ones(self.max_caption_length).unsqueeze(0)

        diminish_cnt = 0
        diminish_pos = []
        i = 0
        while i < ids.shape[1]-4:
            if (ids[0][i] == template1[0]) and (ids[0][i+1] == template1[1]) and (ids[0][i+2] == template1[2] and ids[0][i+3] == template1[3] and ids[0][i+4] == template1[4]):
                self.weights[0][i] = 0.05
                self.weights[0][i+1] = 0.05
                self.weights[0][i+2] = 0.05
                self.weights[0][i+3] = 0.05
                self.weights[0][i+4] = 0.05
                diminish_pos.append(i)
                diminish_pos.append(i+1)
                diminish_pos.append(i+2)
                diminish_pos.append(i+3)
                diminish_pos.append(i+4)
                diminish_cnt += 5
                i += 5
            elif (ids[0][i] == template2[0]) and (ids[0][i+1] == template2[1]) and (ids[0][i+2] == template2[2]):
                self.weights[0][i] = 0.05
                self.weights[0][i+1] = 0.05
                self.weights[0][i+2] = 0.05
                diminish_pos.append(i)
                diminish_pos.append(i+1)
                diminish_pos.append(i+2)
                diminish_cnt += 3
                i += 3
            else:
                i += 1

        masked_ids, mask_pos = self._context_mask(ids)

        mask_diminish = list(filter(lambda x: x in diminish_pos, mask_pos))
        len_dm = len(mask_diminish)
        mask_cnt = len(mask_pos)

        if mask_cnt > 0 and diminish_cnt > 0:
            expand_weight = (0.95 * (diminish_cnt - len_dm) + mask_cnt) / (mask_cnt - 0.95 * len_dm)
            for i in mask_pos:
                self.weights[0][i] = self.weights[0][i] * expand_weight
        elif diminish_cnt > 0:
            expand_weight = self.max_caption_length / (self.max_caption_length - 0.95 * diminish_cnt)
            self.weights[0] = self.weights[0] * expand_weight

        return image, ids, attention_mask, type_ids, masked_ids, self.weights

    def read_csv(self):
        csv_path = os.path.join(self.data_root, 'trainingZero_3090.csv')
        df = pd.read_csv(csv_path, sep=',', dtype={"img_path": str, "report": str, "llm_output": str})
        return df["image_path"], df["report_content"], df["final_report"], [], []  # 移除 attn_i 和 attn_j
        # image_path,report_content,final_report

    def collate_fn(self, instances: List[Tuple]):
        image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list, weights_list = [], [], [], [], [], []
        for b in instances:
            image, ids, attention_mask, type_ids, masked_ids, weights = b
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)
            weights_list.append(weights)

        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()
        weights_stack = torch.stack(weights_list).squeeze()

        return_dict = {
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack,
            "weights": weights_stack
        }
        return return_dict

