import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IntentRNN(nn.Module):
    def __init__(self, 
            input_size=50,
            vocabulary_size=1000,
            hidden_size=50,
            output_intent_size=17,
            num_layers=1,
            bidirectional=False
        ):
        super().__init__()
        self.D = 2 if bidirectional else 1 
        self.embed = nn.Embedding(vocabulary_size, input_size)
            
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional)
        self.intent_clf = nn.Linear(self.D * hidden_size, output_intent_size)
        self.h_init = nn.Parameter(torch.randn(num_layers  * self.D, hidden_size))
        self.c_init = nn.Parameter(torch.randn(num_layers * self.D, hidden_size))
    
    ''' RNN forward
        
	x : A tensor N x S
	seq_len : a list containing the sequence size for each element of the batch
    '''
    def forward(self, x, seq_len):
        N, S = x.shape[0], x.shape[1]
        mask = pad_sequence([torch.ones(i) for i in seq_len], padding_value=0, batch_first=True)

        ### process embedding
        embedding = self.embed(x.long())

        ### ensure that no grad on the padded part of the sequences
        x = pack_padded_sequence(embedding, seq_len, batch_first=True, enforce_sorted=False)
        
        ### forward through the rnn
        x_out, _ = self.rnn(x, 
		                (
		                self.h_init.unsqueeze(1).repeat(1, N, 1),
		                self.c_init.unsqueeze(1).repeat(1, N, 1)
		                )
		           )
        
        x_out, _ = pad_packed_sequence(x_out, batch_first=True)
        

        intent_x_out = torch.einsum('ijk, ij -> ik',x_out, mask)
        intent_x_out = torch.einsum('ik, i -> ik', intent_x_out, 1./mask.sum(-1))

        
        ### apply intent classifier 
        intent_pred = self.intent_clf(intent_x_out)
        
        return {"y_intent": intent_pred}


class IntentSlotRNN(nn.Module):
    def __init__(self, 
            input_size=50,
            hidden_size=50,
            output_intent_size=17,
            output_slot_size=101,
            num_layers=1,
            bidirectional=False
        ):
        super().__init__()
        self.D = 2 if bidirectional else 1 
            
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional)
        self.intent_clf = nn.Linear(self.D * hidden_size, output_intent_size)
        self.slot_clf = nn.Linear(self.D * hidden_size, output_slot_size)
        self.h_init = nn.Parameter(torch.randn(num_layers  * self.D, hidden_size))
        self.c_init = nn.Parameter(torch.randn(num_layers * self.D, hidden_size))
    
    ''' RNN forward
        input : dict
            batched dat
    '''
    def forward(self, input_embeds, seq_len, mask=None):
        ### ensure that no grad on the padded part of the sequences
        N, S = input_embeds.shape[0], input_embeds.shape[1]
        x = torch.nn.utils.rnn.pack_padded_sequence(input_embeds, seq_len, batch_first=True, enforce_sorted=False)
        
        ### forward through the rnn
        x_out, (h_out, c_out) = self.rnn(x, 
                                    (
                                    self.h_init.unsqueeze(1).repeat(1, N, 1),
                                    self.c_init.unsqueeze(1).repeat(1, N, 1)
                                    )
                               )
        
        x_out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_out, batch_first=True)
        
        ### if a mask is given then applies it to compute the mean output vector for intent classification
        if(mask is not None):
            intent_x_out = torch.einsum('ijk, ij -> ik',x_out, mask)
            intent_x_out = torch.einsum('ik, i -> ik', intent_x_out, 1./mask.sum(-1))
        else :
            intent_x_out = x_out.mean(1)
        
        ### apply intent classifier 
        intent_pred = self.intent_clf(intent_x_out)
        slot_pred = self.slot_clf(x_out)
        
        return {"y_intent": intent_pred, "y_slot": slot_pred}



class Seq2SeqRNN(nn.Module):
    def __init__(self, 
            input_size=50,
            hidden_size=50,
            output_vocab_size=17,
            num_layers=1,
            bidirectional=False
        ):
        super().__init__()
        self.D = 2 if bidirectional else 1 
            
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           	   batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           	   batch_first=True, bidirectional=False)
        self.vocab_classifier = nn.Linear(self.D * hidden_size, output_vocab_size)
        self.h_init = nn.Parameter(torch.randn(num_layers  * self.D, hidden_size))
        self.c_init = nn.Parameter(torch.randn(num_layers * self.D, hidden_size))
    
    ''' RNN forward
        input : dict
            batched dat
    '''
    def forward(self, input_embeds, output_embeds, input_seq_len, output_seq_len, mask=None):
        ### ensure that no grad on the padded part of the sequences
        N, S = input_embeds.shape[0], input_embeds.shape[1]
        x = torch.nn.utils.rnn.pack_padded_sequence(input_embeds, input_seq_len, batch_first=True, enforce_sorted=False)
        
        ### forward through the rnn
        x_out, (h_out, c_out) = self.encoder(x, 
                                    (
                                    self.h_init.unsqueeze(1).repeat(1, N, 1),
                                    self.c_init.unsqueeze(1).repeat(1, N, 1)
                                    )
                               )
         
        
        x_out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_out, batch_first=True)
        
        ### get last ouput
        encoded_sequence = x_out[torch.arange(N), input_seq_len]
        y = torch.nn.utils.rnn.pack_padded_sequence(output_embeds, output_seq_len, batch_first=True, enforce_sorted=False)
        y_out, (h_out, c_out) = self.encoder(x, 
                                    (
                                    encoded_sequence,
                                    encoded_sequence
                                    )
                               )  
        y_out, _ = torch.nn.utils.rnn.pad_packed_sequence(y_out, batch_first=True)
        ### if a mask is given then applies it to compute the mean output vector for intent classification
        output = self.vocab_classifier(y_out)
        
        return {"output": output}
    def generate_one(self, input_embeds, seq_max=50, start_token="BOS",input_embedder=None):
        N, S = input_embeds.shape[0], input_embeds.shape[1]
        x = torch.nn.utils.rnn.pack_padded_sequence(input_embeds, input_seq_len, batch_first=True, enforce_sorted=False)
        
        ### forward through the rnn
        x_out, (h_out, c_out) = self.encoder(x, 
                                    (
                                    self.h_init.unsqueeze(1).repeat(1, N, 1),
                                    self.c_init.unsqueeze(1).repeat(1, N, 1)
                                    )
                                )
            
        
        x_out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_out, batch_first=True)
        
        ### get last ouput
        encoded_sequence = x_out[:, -1]
        curr_c = encoded_sequence
        curr_h = encoded_sequence
        seq = []
        x_out, (h_out, c_out) = self.encoder(x, 
                                    (
                                    curr_h,
                                    curr_c
                                    )
                                )
        
        for i in range(seq_max):
            x = input_embedder(start_token)
            
            seq.append(self.vocab_classifier(x_out).argmax())
        return seq
		     

