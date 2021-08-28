from transformers import AutoTokenizer, AutoModelWithLMHead, top_k_top_p_filtering
import torch
from torch.nn import functional as F


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    ids = model.generate(do_sample=True, top_p=0.8)

    sequence = "I am the "
    input_ids = tokenizer.encode(text=sequence, return_tensors="pt")
    # get logits of last hidden state
    next_token_logits = model(input_ids)[0][:, -1, :]
    # filter
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    # sample
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat([input_ids, next_token], dim=-1)
    resulting_string = tokenizer.decode(generated.tolist()[0])
    print("next_token: {}".format(tokenizer.decoder[int(next_token.numpy())]) )
    print(resulting_string)
    print(tokenizer.decode(ids))
