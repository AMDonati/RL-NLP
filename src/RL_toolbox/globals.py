try:
    from vilbert.task_utils import compute_score_with_logits
    from vilbert.vilbert import VILBertForVLTasks, BertConfig
except ImportError:
    print("VILBERT NOT IMPORTED!!")
from transformers import AutoModelWithLMHead, AutoTokenizer

vilbert_path = "output/vilbert_vqav2/model.bin"
bert_config = BertConfig.from_json_file("output/vilbert_vqav2/bert_base_6layer_6conect.json")
vilbert_model = VILBertForVLTasks.from_pretrained(vilbert_path, config=bert_config, num_labels=1)

gpt2_model = AutoModelWithLMHead.from_pretrained("cache/gpt-2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("cache/gpt-2")

