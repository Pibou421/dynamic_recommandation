from torch import nn
import torch, torchtext
from torchtext.data.utils import ngrams_iterator


transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
print(out.shape)

token_list = list('arbre')  # ['here', 'we', 'are']
token_list = "this is a test"
#print(list(ngrams_iterator(token_list, 1)))
from torchtext.data.utils import get_tokenizer

tokenizer = ""
#print(get_tokenizer(token_list))

TEXT = torchtext.data.Field(tokenize=list,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train = ["arbre".zfill(7), "voiture", "maladie", "soutien"]
TEXT.build_vocab(train)
print(TEXT.numericalize(train))
print('long', len(TEXT.vocab.stoi))

#ex = torchtext.data.Example.fromCSV('~/Téléchargements/dico.txt')
#print(ex)
