import argparse
import logging
import os
import pickle

import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader
from vilbert.datasets import VQAClassificationDataset
from vilbert.task_utils import compute_score_with_logits
from vilbert.vilbert import VILBertForVLTasks, BertConfig

from data_provider._image_features_reader import ImageFeaturesH5Reader
from data_provider.vqa_dataset import _load_dataset

logger = logging.getLogger(__name__)


class VQADataset(VQAClassificationDataset):
    def __init__(
            self,
            task,
            dataroot,
            annotations_jsonpath,
            split,
            image_features_reader,
            gt_image_features_reader,
            tokenizer,
            bert_model,
            clean_datasets,
            padding_index=0,
            max_seq_length=16,
            max_region_num=101,
    ):

        # Dataset.__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = pickle.load(open(ans2label_path, "rb"))
        self.label2ans = pickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        clean_train = "_cleaned" if clean_datasets else ""
        cache_path = os.path.join(
            dataroot,
            "cache",
            task + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl",
        )
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split, clean_datasets)
            image_ids = list(map(int, images_feature_reader._image_ids[:-1]))
            self.entries = [entry for entry in self.entries if entry["image_id"] in image_ids]
            self.tokenize(max_seq_length)
            self.tensorize()
            pickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = pickle.load(open(cache_path, "rb"))

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        # shuffle the image location here.
        # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
        # img_idx.append(0)
        # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
        # mix_features_pad[:mix_num_boxes] = features[img_idx]

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, help="path for vilbert")
    parser.add_argument("-config", type=str, help="path for vilbert config")
    parser.add_argument("-features_path", type=str, help="path for features_h5path ")
    parser.add_argument("-dataroot", type=str, help="path for features_h5path",
                        default="/Users/guillaumequispe/PycharmProjects/RL-NLP/data/vqa/VQA")

    args = parser.parse_args()
    config = BertConfig.from_json_file(args.config)
    model = VILBertForVLTasks.from_pretrained(args.path, config=config, num_labels=1)
    images_feature_reader = ImageFeaturesH5Reader(args.features_path, False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dataset = VQADataset(task='VQA',
                         dataroot=args.dataroot,
                         annotations_jsonpath='',
                         split="trainval",
                         image_features_reader=images_feature_reader,
                         gt_image_features_reader=None,
                         tokenizer=tokenizer,
                         bert_model="bert-base-uncased",
                         clean_datasets=True,
                         padding_index=0,
                         max_seq_length=23,
                         max_region_num=101,
                         )
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    print("END")

    for i_batch, sample_batched in enumerate(dataloader):
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
            sample_batched
        )
        batch_size = features.size(0)
        task_tokens = question.new().resize_(question.size(0), 1).fill_(1)
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )
        score = compute_score_with_logits(vil_prediction, target).sum()
        score /= batch_size
        print("{},{}".format(torch.argmax(vil_prediction, dim=-1), torch.argmax(target, dim=-1)))
        _, sorted_indices = torch.sort(vil_prediction, descending=True)
        ranks = torch.gather(sorted_indices, 1, torch.argmax(target, dim=-1).view(-1, 1))
        print("rank {}".format(ranks))
        print("score {}".format(score))
