import torch
import torch.nn as nn
from torch.utils.data import Dataset

class bilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_lang):
        super().__init__()
        self.seq_len=seq_lang
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index:int )  :
        start_idx = index
        end_idx = start_idx + 10
        future_end_idx = end_idx + 10 + 1


        past_prices = self.ds[start_idx:end_idx]
    # Target sequence (next prices)
        future_prices = self.ds[end_idx:future_end_idx]
       # print("past_prices", past_prices)
        future_prices = [str(i) for i in future_prices]#convert to ste
        past_prices = [str(i) for i in past_prices]
        #print("future_prices", future_prices)
        future_prices_str = " ".join(future_prices)#convert to 1 strwith space
        past_prices_str = " ".join(past_prices)
        
    
    # Convert to strings
       

       # print("OOOOOOOOOOOOOOOOOO")
        enc_input_token = self.tokenizer_src.encode(past_prices_str).ids
       # print("enc_input_token", enc_input_token)
        dec_input_token = self.tokenizer_tgt.encode(future_prices_str).ids
        label = self.tokenizer_tgt.encode(future_prices_str).ids
       # print("dec_input_token", dec_input_token)
       # print("eeeeeeeeeeeeeeeeeeeeee")
        enc_num_padding_tokens = self.seq_len - len(enc_input_token) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_token) - 1
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

    # Add special tokens and padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_token, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


    # Convert to tensors
        encoder_input = torch.tensor(encoder_input, dtype=torch.int64)
        decoder_input = torch.tensor(decoder_input, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@22")
        #print("encoder_input", encoder_input)
        #print("decoder_input", decoder_input)
        #print("label", label)   
        #print("encoder_mask", (encoder_input != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int())
        #print("decoder_mask", (decoder_input != self.tokenizer_tgt.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)))
       # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'label': label,
            'encoder_mask': (encoder_input != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.tokenizer_tgt.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
    }
def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
