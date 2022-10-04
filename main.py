# https://huggingface.co/tuner007/pegasus_paraphrase

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text




if __name__ == '__main__':

  context = """
  Hello, and wellcome to autodox I will expalin lots of things here and I hope you will like it as well
  """
  from sentence_splitter import SentenceSplitter, split_text_into_sentences
  splitter = SentenceSplitter(language='en')
  sentence_list = splitter.split(context)
  paraphrase = []
  for i in sentence_list:
    a = get_response(i, 1)
    paraphrase.append(a)

  paraphrase2 = [' '.join(x) for x in paraphrase]
  paraphrase3 = [' '.join(x for x in paraphrase2)]
  paraphrased_text = str(paraphrase3).strip('[]').stripnt(context)
  print(paraphrased_text)

