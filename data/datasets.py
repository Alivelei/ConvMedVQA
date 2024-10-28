# _*_ coding: utf-8 _*_

"""
    @Time : 2023/2/5 11:50 
    @Author : smile 笑
    @File : datasets.py
    @desc :
"""


from torch.utils.data import Dataset
import torch
import re
import json
import pickle
import os
import torchvision.transforms as tfs
from PIL import Image
from data.word_sequence import sentence_to_word, word_id_transform
import numpy as np
import random


def train_aug_img(img, args, img_mean, img_std):
    if args.general_rand_aug:
        aug = tfs.Compose([
            tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_scale_left, args.resized_crop_scale_right),
                                  ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(p=args.img_flip),
            tfs.RandAugment(args.ra_n, args.ra_m),
            tfs.ColorJitter(args.img_jitter, args.img_jitter, args.img_jitter),
            tfs.ToTensor(),
            tfs.Normalize(img_mean, img_std),
            tfs.RandomErasing(args.reprob)
        ])
    else:
        aug = tfs.Compose([
            tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_left, args.resized_crop_right)),
            tfs.RandomApply([tfs.GaussianBlur(kernel_size=args.b_size, sigma=args.blur)], p=args.blur_p),
            tfs.RandomGrayscale(p=args.grayscale),
            tfs.RandomApply([
                tfs.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
                p=args.apply_p
            ),
            tfs.RandomRotation(args.img_rotation),
            tfs.RandomHorizontalFlip(args.img_flip),
            tfs.ToTensor(),
            tfs.Normalize(img_mean, img_std)
        ])

    return aug(img)


def test_aug_img(img, args, img_mean, img_std):
    aug = tfs.Compose([
        tfs.Resize([args.img_height, args.img_width]),
        tfs.ToTensor(),
        tfs.Normalize(img_mean, img_std)
    ])

    return aug(img)


class SlakeDatasetModule(Dataset):
    def __init__(self, args, dataset_path, mode):
        self.args = args

        self.mode = mode
        self.xm_path = args.slake_dataset_xm_path
        self.queries = json.load(open(dataset_path, encoding="utf-8"))

        # 只取英文部分
        self.queries = [query for query in self.queries if query["q_lang"] == "en"]  # 4919、1061

        if mode == "train":
            dataset_split_rate = args.dataset_split_rate
            if dataset_split_rate < 1:
                self.queries = random.sample(self.queries, int(len(self.queries)*dataset_split_rate)+1)

        if args.use_pretrain:
            self.qus_ws = pickle.load(open(args.all_word_ws_path, "rb"))
        else:
            self.qus_ws = pickle.load(open(args.slake_qus_ws_path, "rb"))
        self.ans_ws = pickle.load(open(args.slake_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

        self.slake_img_mean = args.slake_img_mean
        self.slake_img_std = args.slake_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.xm_path + str(query["img_id"]), "source.jpg")

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(query["answer"], False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.slake_img_mean, self.slake_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.slake_img_mean, self.slake_img_std)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        qus_id = word_id_transform(self.qus_ws, question, max_len=self.max_seq_len)
        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, qus_id, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class RadDatasetModule(Dataset):
    def __init__(self, args, rad_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.rad_images_path
        self.queries = json.load(open(rad_dataset_path, encoding="utf-8"))

        if mode == "train":
            dataset_split_rate = args.dataset_split_rate
            if dataset_split_rate < 1:
                self.queries = random.sample(self.queries, int(len(self.queries)*dataset_split_rate)+1)

        if args.use_pretrain:
            self.qus_ws = pickle.load(open(args.all_word_ws_path, "rb"))
        else:
            self.qus_ws = pickle.load(open(args.rad_qus_ws_path, "rb"))
        self.ans_ws = pickle.load(open(args.rad_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

        self.rad_img_mean = args.rad_img_mean
        self.rad_img_std = args.rad_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]
        img_path = os.path.join(self.images_path, str(query["image_name"]))

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(str(query["answer"]), False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.rad_img_mean, self.rad_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.rad_img_mean, self.rad_img_std)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        qus_id = word_id_transform(self.qus_ws, question, max_len=self.max_seq_len)
        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, qus_id, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class CLEFDatasetModule(Dataset):
    def __init__(self, args, dataset_img_path, dataset_json_path, mode):
        self.args = args
        self.mode = mode

        self.dataset_img_path = dataset_img_path

        self.queries = json.load(open(dataset_json_path, encoding="utf-8"))

        if mode == "train":
            dataset_split_rate = args.dataset_split_rate
            if dataset_split_rate < 1:
                self.queries = random.sample(self.queries, int(len(self.queries)*dataset_split_rate)+1)

        if args.use_pretrain:
            self.qus_ws = pickle.load(open(args.all_word_ws_path, "rb"))
        else:
            self.qus_ws = pickle.load(open(args.clef_qus_ws_path, "rb"))

        self.ans_ws = pickle.load(open(args.clef_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

        self.clef_sort = args.clef_sort

        self.clef_img_mean = args.clef_img_mean
        self.clef_img_std = args.clef_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]  # 随机抽取一个

        img_path = os.path.join(self.dataset_img_path, str(query["img_path"]+".jpg"))

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(query["answer"], False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.clef_img_mean, self.clef_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.clef_img_mean, self.clef_img_std)

        category = query["category"]
        category_id = self.clef_sort[str(category)]

        qus_id = word_id_transform(self.qus_ws, question, max_len=self.max_seq_len)
        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, qus_id, ans_id, category_id

    def __len__(self):
        return len(self.queries)


class PathVQADatasetModule(Dataset):
    def __init__(self, args, img_folder_path, dataset_text_path, mode):
        self.args = args
        self.mode = mode
        self.img_folder_path = img_folder_path

        self.queries = pickle.load(open(dataset_text_path, "rb"))

        if mode == "train":
            dataset_split_rate = args.dataset_split_rate
            if dataset_split_rate < 1:
                self.queries = random.sample(self.queries, int(len(self.queries)*dataset_split_rate)+1)

        self.max_seq_len = args.qus_seq_len
        if args.use_pretrain:
            self.qus_ws = pickle.load(open(args.all_word_ws_path, "rb"))
        else:
            self.qus_ws = pickle.load(open(args.path_vqa_qus_ws_path, "rb"))[0]
        self.ans_ws = pickle.load(open(args.path_vqa_ans_ws_path, "rb"))

        self.path_vqa_img_mean = args.path_vqa_img_mean
        self.path_vqa_img_std = args.path_vqa_img_std

    def word_2id(self, question, dic, max_seq_len=20):
        sentence = [i for i in re.findall("[a-z0-9]*", question.lower()) if len(i) > 0]

        if max_seq_len is not None:
            if max_seq_len > len(sentence):
                sentence = sentence + ["<unk>"] * (max_seq_len - len(sentence))
            if max_seq_len < len(question):
                sentence = sentence[:max_seq_len]
        return [int(dic.get(word, dic["<unk>"])) for word in sentence]

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.img_folder_path, query["image"] + ".jpg")

        answer = query["answer"]

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.path_vqa_img_mean, self.path_vqa_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.path_vqa_img_mean, self.path_vqa_img_std)

        if answer.lower() == "no" or answer.lower() == "yes":
            answer_type_id = self.args.answer_close
        else:
            answer_type_id = self.args.answer_open

        if self.args.use_pretrain:
            sentence = sentence_to_word(query["question"])
            qus_id = word_id_transform(self.qus_ws, sentence, max_len=self.max_seq_len)
        else:
            qus_id = self.word_2id(query["question"], self.qus_ws, self.max_seq_len)

        ans_id = self.ans_ws.get(answer, self.ans_ws["unknown"])

        return image, qus_id, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class OVQADatasetModule(Dataset):
    def __init__(self, args, ovqa_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.ovqa_images_path
        self.queries = json.load(open(ovqa_dataset_path, encoding="utf-8"))

        if args.use_pretrain:
            self.qus_ws = pickle.load(open(args.all_word_ws_path, "rb"))
        else:
            self.qus_ws = pickle.load(open(args.ovqa_qus_ws_path, "rb"))
        self.ans_ws = pickle.load(open(args.ovqa_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

        self.ovqa_img_mean = args.ovqa_img_mean
        self.ovqa_img_std = args.ovqa_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.images_path, str(query["image_name"]))

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(str(query["answer"]), False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.ovqa_img_mean, self.ovqa_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.ovqa_img_mean, self.ovqa_img_std)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        qus_id = word_id_transform(self.qus_ws, question, max_len=self.max_seq_len)
        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, qus_id, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)



