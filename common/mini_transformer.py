import torch
import torch.nn as nn 

class PEMethod(nn.Module):
      """Base class — no-op defaults for all three hooks."""
      def apply_to_input(self, token_emb):
          return token_emb                

      def apply_to_qk(self, q, k):
          return q, k                   

      def apply_to_scores(self, scores):
          return scores    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False , max_len=2048 , pe_method=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        self.causal = causal
        self.pe_method = pe_method 

        self.qkv_projection = nn.Linear(d_model,d_model*3)
        self.out_projection = nn.Linear(d_model,d_model)

        self.register_buffer("causal_mask",torch.triu(torch.ones(max_len,max_len,dtype=torch.bool),diagonal=1))

    def forward(self, x):
        # x -> bs,seq_len,d_model
        bs,seq_len,_ = x.shape 

        Q,K,V = torch.chunk(self.qkv_projection(x),3,dim=-1) # bs,seq_len,d_model

        Q = Q.view(bs,seq_len,self.n_heads,self.d_head).permute(0,2,1,3) #bs,n_heads,seq_len,d_head
        K = K.view(bs,seq_len,self.n_heads,self.d_head).permute(0,2,1,3) #bs,n_heads,seq_len,d_head
        V = V.view(bs,seq_len,self.n_heads,self.d_head).permute(0,2,1,3) #bs,n_heads,seq_len,d_head

        ## IF ROPE
        Q,K = self.pe_method.apply_to_qk(Q,K)
        scores = Q@K.transpose(-2,-1) / self.d_head**0.5

        ## IF ALIBI
        scores = self.pe_method.apply_to_scores(scores)

        if self.causal:
            scores = scores.masked_fill(self.causal_mask[:seq_len,:seq_len],float("-inf"))

        weights = torch.softmax(scores,dim=-1)
        out = weights@V

        out = out.permute(0,2,1,3).contiguous().view(bs,seq_len,self.d_model)
        out = self.out_projection(out)

        return out
    

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads , causal, max_len , pe_method):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            causal=causal,
            max_len=max_len,
            pe_method = pe_method
        )

        self.norm_1 = nn.RMSNorm(d_model)
        self.norm_2 = nn.RMSNorm(d_model)

        self.ffn = nn.Sequential(
          nn.Linear(d_model,d_model*4),
          nn.GELU(),
          nn.Linear(d_model*4,d_model)
        )
    
    def forward(self,x):
        # x -> bs,seq_len,d_model
        #Pre-norm
        h = self.norm_1(x)
        # attention
        h = self.attention(h)
        # residual
        h = x + h
        # Norm before FFN
        out = self.norm_2(h)
        # FFN
        out = self.ffn(out)
        # Residual
        out = h + out
        
        return out


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, pe_method, causal):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        self.pe_method = pe_method if pe_method is not None else PEMethod()

        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(
                d_model,
                n_heads,
                causal,
                max_len,
                self.pe_method) for _ in range(n_layers)
        ])
        
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        # x -> bs,seq_len
        x = self.token_embedding(x) # bs,seq_len,d_model
        x = self.pe_method.apply_to_input(x)
        
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.final_norm(x)
        
        out = self.lm_head(x)

        return out

        
    
