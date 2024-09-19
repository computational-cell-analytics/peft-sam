# Finetuning experiments on LIVECell data

Finetuning SAM on LIVECell with 50k iterations
- using full finetuning
- using low rank adaption (LoRA)
- freezing the image encoder
- using factor tuning (FacT)


|                    |         ais |       amg |   single box |   single point |   iterative_prompts_start_box |   iterative_prompts_start_point |
|:-------------------|------------:|----------:|-------------:|---------------:|------------------------------:|--------------------------------:|
| Full Finetuning    |   0.420385  |  0.325352 |     0.669954 |     0.461808   |         0.835390              |         0.795713                |
| LoRA               |   0.393673  |  0.273875 |     0.653592 |     0.423316   |         0.830612              |         0.788462                |
| FacT               |   0.387581  |  0.262546 |     0.644785 |     0.406561   |         0.824258              |         0.778803                |
| Frozen Encoder     |   0.357978  |  0.195104 |     0.597674 |     0.325612   |         0.805326              |         0.748055                |
