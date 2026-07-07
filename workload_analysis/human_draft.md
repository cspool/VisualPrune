forward 1

layer 0: attn weights完整计算， v@v=0，tail@v计算sum（稀疏）， 但SV计算没有跳过。

layer （1～）5：attn weights完整计算， tail@v=0（稀疏）。

layer 7(~17)：最后一个token的注意力激活作为ref。 每个visual token(v)对最后一个token激活ref的贡献c，是最后一个token对v的注意力 * v对应的V向量。每个v的delta计算为ref-c。

layer 18: 同layer 7～17计算delta， 再计算每个v的score=(-c)^2（这里说明基于ref的delta似乎是多余的计算？？？），之后比较阈值来决定v token是否留下，记录留下的v token的序列index（10个v token）。

layer 19（～27）：输入序列裁减到58个token（35 text + 10 v + 13 tail），其余同layer 7～17计算， 同样的和ref作差计算delta(?)。

layer 28： 输入完全移除v tyokens，其余同layer 7～17计算， 但不需要计算visual delta。


forward 2：

layer 18： 上一轮的输出token计算， Attn算子复用KV Cache（624个tokens）计算。

layer 19（～27）： 上一轮的输出token计算， Attn算子复用KV Cache（58个tokens）计算。

layer 28（～31）： 上一轮的输出token计算， Attn算子复用KV Cache（48个tokens）计算。


forward 32：

layer 18：上一轮的输出token计算， Attn算子复用KV Cache（624 + 30 = 654个tokens）计算。

layer 19（～27）：上一轮的输出token计算， Attn算子复用KV Cache（58 + 30 = 88个tokens）计算。

layer 28（～31）： 上一轮的输出token计算， Attn算子复用KV Cache（48 + 30 = 78个tokens）计算。