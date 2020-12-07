import torch
import json
import _pickle as cPickle
import os
import numpy as np

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def assert_correctness_batch(inputs, targets):
    assert torch.all(torch.eq(inputs[1:], targets[:-1])) == 1, "error in inputs/targets"


def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def split_question(question):
    last_id = question.nonzero()[-1]
    input_question = torch.cat([question[:last_id], question[last_id + 1:]])
    target_question = question[1:]
    return input_question, target_question


def clean_key(key, tokens_to_remove):
    bool = False
    for tok in tokens_to_remove:
        if tok in key:
            bool = True
            if tok == "?" and key == "?":
                bool = False
    return bool


def get_questions_answers(dataroot, name):
    question_path = os.path.join(
        dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name)
    questions = sorted(
        json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
    )
    answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
    answers = cPickle.load(open(answer_path, "rb"))
    answers = sorted(answers, key=lambda x: x["question_id"])
    return questions, answers


def _load_dataset(dataroot, name, clean_datasets):
    """Load entries
    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """
    if name == "train" or name == "val":
        questions, answers = get_questions_answers(dataroot, name)

    elif name == "trainval":
        questions_train, answers_train = get_questions_answers(dataroot, "train")
        questions_val, answers_val = get_questions_answers(dataroot, "val")
        questions = questions_train + questions_val
        answers = answers_train + answers_val

    elif name == "minval":
        questions_val, answers_val = get_questions_answers(dataroot, "val")
        questions = questions_val[-5000:]
        answers = answers_val[-5000:]

    elif name == "mintrain":
        questions_train, answers_train = get_questions_answers(dataroot, "train")
        questions = questions_train[60000:80000]
        answers = answers_train[60000:80000]

    elif name == "mintrainval":
        questions_val, answers_val = get_questions_answers(dataroot, "val")
        questions_val = questions_val[-5000:]
        answers_val = answers_val[-5000:]
        questions_train, answers_train = get_questions_answers(dataroot, "train")
        questions_train = questions_train[60000:80000]
        answers_train = answers_train[60000:80000]
        questions = questions_train + questions_val
        answers = answers_train + answers_val

    elif name == "test":
        question_path_test = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % "test"
        )
        questions_test = sorted(
            json.load(open(question_path_test))["questions"],
            key=lambda x: x["question_id"],
        )
        questions = questions_test
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    else:
        assert_eq(len(questions), len(answers))
        entries = []
        remove_ids = []
        if clean_datasets:
            remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
            remove_ids = [int(x) for x in remove_ids]
        for question, answer in zip(questions, answers):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(_create_entry(question, answer))

    return entries