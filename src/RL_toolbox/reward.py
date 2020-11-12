import json

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.vilbert import VILBertForVLTasks, BertConfig
from vr.utils import load_execution_engine, load_program_generator


def get_vocab(key, vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)[key]
    return vocab


class Reward:
    def __init__(self, path):
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get(self, question, ep_questions_decoded):
        pass


class Cosine(Reward):
    def __init__(self, path, vocab=None, dataset=None):
        Reward.__init__(self, path)
        with open(path) as json_file:
            data = json.load(json_file)
        df = pd.read_json(json.dumps(data["questions"]))
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(df.question)

    def get(self, question, ep_questions_decoded):
        data = np.append(np.array([question]), np.array(ep_questions_decoded))
        X = self.vectorizer.transform(data)
        S = cosine_similarity(X[0:1], X[1:])
        rew = max(S[0])
        return rew


class Levenshtein_(Reward):
    def __init__(self, path=None, vocab=None, dataset=None):
        Reward.__init__(self, path)
        self.type = "episode"

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if question is None:
            return 0., "N/A"
        distances = np.array([nltk.edit_distance(question.split(), true_question.split()) for true_question in
                              ep_questions_decoded])
        reward = -min(distances) if done else 0
        return reward, ep_questions_decoded[distances.argmin()], None


class Bleu(Reward):
    def __init__(self, path=None, vocab=None, dataset=None):
        Reward.__init__(self, path)
        self.type = "episode"

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        reward = sentence_bleu(list(map(str.split, ep_questions_decoded)), question.split())
        return reward, "N/A", None


class Differential(Reward):
    def __init__(self, reward_function, path=None, vocab=None, dataset=None):
        Reward.__init__(self, path)
        self.type = "step"
        self.reward_function = reward_function
        self.last_reward = None

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if step_idx == 0:
            self.last_reward, _ = self.reward_function.get("", ep_questions_decoded, step_idx=step_idx, done=True)
        reward, closest_question, pred_answer = self.reward_function.get(question, ep_questions_decoded,
                                                                         step_idx=step_idx,
                                                                         done=True)
        diff_reward = reward - self.last_reward
        self.last_reward = reward
        return diff_reward, closest_question, pred_answer


class VQAAnswer(Reward):
    def __init__(self, path=None, vocab=None, dataset=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.execution_engine, ee_kwargs = load_execution_engine(path)
        self.execution_engine.to(self.device)
        self.execution_engine.eval()
        self.program_generator, pg_kwargs = load_program_generator(path)
        self.program_generator.to(self.device)
        self.program_generator.eval()
        self.vocab = vocab
        self.dataset = dataset
        self.vocab_questions_vqa = get_vocab('question_token_to_idx', self.vocab)
        # self.vocab_questions_vqa.update({"<pad>": 0, "<sos>": 1, "<eos>": 2})
        self.trad_dict = {value: self.vocab_questions_vqa[key] for key, value in self.dataset.vocab_questions.items() if
                          key in self.vocab_questions_vqa}

    def trad(self, state):
        idx_vqa = [self.trad_dict[idx] for idx in state.text.squeeze().cpu().numpy() if idx in self.trad_dict]
        idx_vqa.insert(0, 1)  # add SOS token.
        idx_vqa.append(2)  # add EOS token.
        return torch.tensor(idx_vqa).unsqueeze(dim=0)

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        with torch.no_grad():
            question = self.trad(state).to(self.device)
            programs_pred = self.program_generator(question)
            scores = self.execution_engine(state.img.to(self.device), programs_pred)
            _, preds = scores.data.cpu().max(1)
            reward = (preds == real_answer).sum().item()
        return reward, "N/A", preds


class VILBERT(Reward):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.task_id = 1
        self.env = env
        config = BertConfig.from_json_file(vocab)  # TODO: find the json file from vilbert-mt github config folder.
        self.model = VILBertForVLTasks.from_pretrained(
            path,
            config=config,
            num_labels=1)

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None, entry=None):
        if not done:
            return 0, "N/A", None
        (features, spatials, image_mask, co_attention_mask, real_question, target, input_mask, segment_ids,
         labels, entry) = self.dataset.last_entry
        encoded_question = self.dataset.reward_tokenizer.encode(question)
        encoded_question = self.dataset.reward_tokenizer.add_special_tokens_single_sentence(encoded_question)
        encoded_question=self.dataset.reward_tokenizer.add_special_tokens_single_sentence(
            list(real_question[real_question != 0].numpy()))
        if type(encoded_question) != torch.tensor:
            encoded_question = torch.tensor(encoded_question).view(-1)
        encoded_question = F.pad(input=encoded_question, pad=(0, real_question.size(0) - encoded_question.size(0)),
                                 mode='constant', value=0)
        task_tokens = encoded_question.new().resize_(encoded_question.size(0), 1).fill_(int(self.task_id))

        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = self.model(
            encoded_question.unsqueeze(dim=0),
            features.unsqueeze(dim=0),
            spatials.unsqueeze(dim=0),
            segment_ids.unsqueeze(dim=0),
            input_mask.unsqueeze(dim=0),
            image_mask.unsqueeze(dim=0),
            co_attention_mask.unsqueeze(dim=0),
            task_tokens
        )
        reward=torch.argmax(vil_prediction) == torch.argmax(target)
        reward=int(reward)
        return reward, "N/A", None


rewards = {"cosine": Cosine, "levenshtein": Levenshtein_, "vqa": VQAAnswer, "bleu": Bleu, "vilbert": VILBERT}

if __name__ == '__main__':
    reward_func = rewards["cosine"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples_old/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} cosine".format(rew))
    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} cosine".format(rew))

    print("levenshtein rewards...")

    reward_func = rewards["levenshtein"](path=None)
    reward_func_ = rewards["levenshtein_"](path=None)

    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    str_1 = "is it blue and tiny ?"
    str_2 = "is it blue ?"
    print('question', str_1)
    print('true question', str_2)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    rew, _ = reward_func_.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
    print('rew not norm', rew)

    str_1 = "is it red ?"
    str_2 = "is it blue ?"
    print('question', str_1)
    print('true question', str_2)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    rew, _ = reward_func_.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
    print('rew not norm', rew)

    print('checking rew with an empty string...')
    rew_norm_pos, sim_q = reward_func.get("", [str_2])
    print('rew norm positive', rew_norm_pos)
    print('similar question', sim_q)

    str_1 = "blue red it"
    str_2 = "is it blue"
    print('question', str_1)
    print('true question', str_2)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)

    # --------------------------- CorrectVocab reward -----------------------------------------------------------------------------
    print("correct vocab reward...")

    reward_func = rewards["correct_vocab"](path=None)
    str_1 = "is it blue ?"
    str_2 = "is it blue ?"
    print(str_1)
    print(str_2)
    # rew = reward_func.get(str_1, [str_2])
    print('reward', rew)

    str_1 = "blue it is ?"
    str_2 = "is it blue ?"
    print(str_1)
    print(str_2)
    rew = reward_func.get(str_1, [str_2, "red is there a black ball ?"])
    print('reward', rew)

    str_1 = "red it is ?"
    str_2 = "is it blue ?"
    print(str_1)
    print(str_2)
    rew = reward_func.get(str_1, [str_2])
    print('reward', rew)

    str_1 = "red that object"
    str_2 = "is it blue"
    print(str_1)
    print(str_2)
    rew = reward_func.get(str_1, [str_2])
    print('reward', rew)

    # ----- lv reward with correct vocab ---------------------------------------------------------------------------------------
    print("lv reward with vocab...")
    reward_func = rewards["levenshtein"](path=None)
    reward_func_vocab = rewards["levenshtein"](path=None, correct_vocab=True)
    ref_questions = ["is it blue", "red is there a black ball"]
    str = "blue it is"
    print('question', str)
    print('ref_questions', ref_questions)
    print('reward w/o vocab', reward_func.get(str, ref_questions))
    print('rew with vocab', reward_func_vocab.get(str, ref_questions))
    str = "blue red it"
    print('question', str)
    print('ref_questions', ref_questions)
    print('reward w/o vocab', reward_func.get(str, ref_questions))
    print('rew with vocab', reward_func_vocab.get(str, ref_questions))
    str = "is it blue"
    print('question', str)
    print('ref_questions', ref_questions)
    print('reward w/o vocab', reward_func.get(str, ref_questions))
    print('rew with vocab', reward_func_vocab.get(str, ref_questions))
