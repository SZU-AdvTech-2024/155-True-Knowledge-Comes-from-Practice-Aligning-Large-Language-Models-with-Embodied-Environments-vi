import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
class CriticWithDropout(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, dropout_prob=0.1):  # 增加dropout_prob参数
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.dropout_prob = dropout_prob  # 保存 dropout 概率为实例属性
        
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
            self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
            self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(self.dropout_prob)  # 增加 Dropout 层
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      use_cache=False):

        with torch.no_grad():
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True)

        hidden_states = transformer_outputs[1][-1][:, -1, :].float()

        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        x = self.dropout(x)  # 在第二层 MLP 后增加 Dropout
        values = self.v_head_mlp3(x).squeeze(-1)

        return values
    
    def predict_with_uncertainty(self,
                                 input_ids=None,
                                 attention_mask=None,
                                 past_key_values=None,
                                 head_mask=None,
                                 inputs_embeds=None,
                                 use_cache=False,
                                 mc_samples=10):  # 新增不确定性量化方法
        """
        Perform Monte Carlo sampling to compute mean and uncertainty.
        """
        all_values = []
        # 确保输入在循环中不被改变
        # input_ids_copy = input_ids.clone()
        # attention_mask_copy = attention_mask.clone()
        # print("one step")
        for _ in range(mc_samples):  # Monte Carlo 采样
            # print(f"input_ids shape: {input_ids.shape}")
            # print(f"attention_mask shape: {attention_mask.shape}")

            values = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
            # print("input_ids shape:", input_ids.shape)
            # print("attention_mask shape:", attention_mask.shape)
            # print("values shape:", values.shape)

            # 验证每次采样的输出形状
            # assert values.shape[0] == input_ids_copy.shape[0], "Shape mismatch in MC sampling"
            all_values.append(values)
        
        all_values = torch.stack(all_values, dim=0)
        assert all_values.shape[1] == input_ids.shape[0], "Stacked values shape mismatch"
        mean_values = all_values.mean(dim=0)  # 计算value均值
        uncertainty = all_values.std(dim=0)  # 计算value标准差
        assert uncertainty.shape == mean_values.shape, "Uncertainty shape mismatch"
        # print("mean_values shape:", mean_values.shape)
        # print("uncertainty shape:", uncertainty.shape)

        return mean_values, uncertainty
    
    
    
# import torch
# from torch import nn

# class CriticWithDropout(nn.Module):  # 修改 1：更改类名，表明引入了 Dropout 和不确定性量化

#     def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, dropout_prob=0.1):  # 修改 2：新增 dropout_prob 参数
#         super().__init__()
#         self.config = base_model.config
#         self.num_padding_at_beginning = num_padding_at_beginning
#         self.dropout_prob = dropout_prob  # 修改 3：保存 dropout 概率为实例属性
        
#         if hasattr(self.config, "word_embed_proj_dim"):
#             # `OPT` models use word_embed_proj_dim as final output
#             self.v_head = nn.Linear(self.config.word_embed_proj_dim,
#                                     1,
#                                     bias=False)
#         else:
#             # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd`
#             self.config.n_embd = self.config.hidden_size if hasattr(
#                 self.config, "hidden_size") else self.config.n_embd
#             self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
#             self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
#             self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
#             self.relu = nn.ReLU()
#             self.dropout = nn.Dropout(self.dropout_prob)  # 修改 4：增加 Dropout 层
        
#         self.rwtranrsformer = base_model
#         self.PAD_ID = tokenizer.pad_token_id

#     def gradient_checkpointing_enable(self):
#         self.rwtranrsformer.gradient_checkpointing_enable()

#     def gradient_checkpointing_disable(self):
#         self.rwtranrsformer.gradient_checkpointing_disable()

#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 past_key_values=None,
#                 head_mask=None,
#                 inputs_embeds=None,
#                 use_cache=False):

#         with torch.no_grad():
#             transformer_outputs = self.rwtranrsformer(
#                 input_ids,
#                 past_key_values=past_key_values,
#                 attention_mask=attention_mask,
#                 use_cache=use_cache,
#                 output_hidden_states=True)

#         hidden_states = transformer_outputs[1][-1][:, -1, :].float()

#         x = self.relu(self.v_head_mlp1(hidden_states))
#         x = self.dropout(self.relu(self.v_head_mlp2(x)))  # 修改 5：在第二层 MLP 后增加 Dropout
#         values = self.v_head_mlp3(x).squeeze(-1)

#         return values

#     def predict_with_uncertainty(self,
#                                  input_ids=None,
#                                  attention_mask=None,
#                                  past_key_values=None,
#                                  head_mask=None,
#                                  inputs_embeds=None,
#                                  use_cache=False,
#                                  mc_samples=10):  # 修改 6：新增不确定性量化方法
#         """
#         Perform Monte Carlo sampling to compute mean and uncertainty.
#         """
#         all_values = []
#         for _ in range(mc_samples):  # 修改 7：Monte Carlo 采样
#             values = self.forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 past_key_values=past_key_values,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 use_cache=use_cache,
#             )
#             all_values.append(values)
        
#         all_values = torch.stack(all_values, dim=0)
#         mean_values = all_values.mean(dim=0)  # 修改 8：计算均值
#         uncertainty = all_values.std(dim=0)  # 修改 9：计算标准差
#         return mean_values, uncertainty

