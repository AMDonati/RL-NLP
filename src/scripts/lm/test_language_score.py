from eval.metric import *

class LanguageScore:
    '''Compute the perplexity of a pretrained language model (GPT) on the generated dialog.'''

    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.lm_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.measure = []
        self.metric = []

    def fill_(self, **kwargs):
        if kwargs["state"].text.shape[-1] > 1:
            state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, 1:].cpu().numpy()[0])
            inputs = self.tokenizer(state_decoded, return_tensors="pt")
            _, logits = self.lm_model(**inputs, labels=inputs["input_ids"])
            scores = F.log_softmax(logits, dim=-1)  # (B, S, vocab size)
            action_decoded = self.dataset.question_tokenizer.decode(kwargs["action"].cpu().numpy())
            action_id = self.tokenizer(action_decoded)
            log_prob = scores[:, -1, action_id["input_ids"][0]]
            self.measure.append(log_prob.squeeze())

    def compute_(self, **kwargs):
        if len(self.measure) > 0:
            ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).detach().cpu().numpy().item()
            self.metric.append(ppl)
            self.measure = []

    def reset(self):
        self.metric = []


if __name__ == '__main__':
    from data_provider.vqa_dataset import VQADataset
    from data_provider.vqa_tokenizer import VQATokenizer
    from data_provider._image_features_reader import ImageFeaturesH5Reader
    from pytorch_transformers import BertTokenizer
    from transformers import GPT2Tokenizer
    from collections import namedtuple
    import torch

    State = namedtuple('State', ('text', 'img', "answer"))
    features_h5path = "../../../data/vqa-v2/coco_trainval.lmdb"
    data_path = "../../../data/vqa-v2"
    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)
    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)
    vocab_path = os.path.join(data_path, "cache", "vocab_min.json")
    dataset = VQADataset(split="mintrain", dataroot=data_path,
                              image_features_reader=images_feature_reader, question_tokenizer=question_tokenizer,
                              reward_tokenizer=reward_tokenizer, clean_datasets=True,
                              max_seq_length=23, min_len_questions=0,
                              num_answers=1, num_images=None, filter_entries=True,
                              vocab_path=vocab_path)

    language_score = LanguageScore(dataset=dataset)

    def get_state_actions(test_sentence):
        state_encoded = dataset.question_tokenizer.encode(test_sentence)
        state_encoded = [1] + state_encoded
        state_encoded = state_encoded[:-1]
        sequence_actions = state_encoded [1:]
        return state_encoded, sequence_actions

    def compute_language_score(state_encoded, sequence_actions):
        for i, action in enumerate(sequence_actions):
            state = torch.tensor(state_encoded[:i+1]).unsqueeze(0)
            state_ = State(state, None, None)
            action_ = torch.tensor(action).view(1)
            language_score.fill_(state=state_, action=action_)
        language_score.compute_()
        return language_score.metric[0]


    print("true sentence case ...............")
    test_sentence = "The sky is blue"
    print(test_sentence)
    state_encoded, actions = get_state_actions(test_sentence)
    score = compute_language_score(state_encoded, actions)
    print("language score on true sentence", score)

    print("true question case ...............")
    test_sentence = "What color is the plane?"
    print(test_sentence)
    state_encoded, actions = get_state_actions(test_sentence)
    score = compute_language_score(state_encoded, actions)
    print("language score on true sentence", score)

    language_score.reset()
    print("shifted sentence case..........")
    test_sentence = "color plane What the red is?"
    print(test_sentence)
    state_encoded, actions = get_state_actions(test_sentence)
    score = compute_language_score(state_encoded, actions)
    print("language score on shifted sentence", score)

    language_score.reset()
    print("drifted sentence case..........")
    test_sentence = "What the the the the the?"
    print(test_sentence)
    state_encoded, actions = get_state_actions(test_sentence)
    score = compute_language_score(state_encoded, actions)
    print("language score on drifted sentence", score)

    language_score.reset()
    print("no sense sentence case..........")
    test_sentence = "toilet cake company tree cows boat"
    print(test_sentence)
    state_encoded, actions = get_state_actions(test_sentence)
    score = compute_language_score(state_encoded, actions)
    print("language score on no sense sentence", score)

    print("end")