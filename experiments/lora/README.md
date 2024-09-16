# Testing LoRA on different light and electron miscroscopy datasets

### Datasets
**gonuclear**: Dataset for nuclei segmentation in 3D fluorescence microscopy (lm)
**orgasegment**: Dataset for the segmentation of human intestinal organoids (lm)  
**covid_if**: Datset that contains annotation for cell and nucleus segmentation in immunofluorescence microscopy(lm)  
**mitolab/glycolytic_muscle**: Dataset for mitochondria segmentation (em)  
**platynereis/cilia**: Dataset for the segmentation of cilia in the platynereis larve. 

## Comparing LoRA with Full Finetuning
On all 5 datasets the SAM model is trained from default and from the micro-sam models using LoRA and full finetuning.  
Note that all datasets except GuNuclear are trained with early stopping while GoNuclear is trained for 50k iterations.

### Orgasegment

|                    |        ais |      amg |   single box |   single point |   iterative_prompts_start_box |   iterative_prompts_start_point |
|:-------------------|-----------:|---------:|-------------:|---------------:|------------------------------:|--------------------------------:|
| generalist         |   0.239504 | 0.313557 |     0.766814 |       0.426106 |                      0.870109 |                        0.813314 |
| specialist_full_ft |   0.51541  | 0.423463 |     0.790572 |       0.556546 |                      0.87495  |                        0.841196 |
| specialist_lora    |   0.478586 | 0.372293 |     0.77696  |       0.510568 |                      0.872069 |                        0.839907 |
| vanilla            | -| 0.343003 |     0.630294 |       0.494892 |                      0.484599 |                        0.475463 |

### Platynereis Cilia


|                    |        ais |        amg |   single box |   single point |   iterative_prompts_start_box |   iterative_prompts_start_point |
|:-------------------|-----------:|-----------:|-------------:|---------------:|------------------------------:|--------------------------------:|
| generalist         |   0        | 0.00875561 |     0.184394 |      0.0593038 |                      0.429399 |                        0.349337 |
| specialist_full_ft |   0.236222 | 0.208035   |     0.371877 |      0.234535  |                      0.524445 |                        0.451413 |
| specialist_lora    |   0.212794 | 0.137063   |     0.369033 |      0.244724  |                      0.51684  |                        0.446328 |
| vanilla            | -| 0.135574   |     0.26555  |      0.175185  |                      0.195464 |                        0.107775 |


### Mitolab Glycolytic Muscle
|                    |        ais |      amg |   single box |   single point |   iterative_prompts_start_box |   iterative_prompts_start_point |
|:-------------------|-----------:|---------:|-------------:|---------------:|------------------------------:|--------------------------------:|
| generalist         |   0.430989 | 0.357421 |     0.791705 |       0.519416 |                      0.931755 |                        0.902345 |
| specialist_full_ft |   0.616114 | 0.552763 |     0.856562 |       0.665076 |                      0.966356 |                        0.946224 |
| specialist_lora    |   0.658877 | 0.53201  |   0.865823|       0.714701 |                    0.9657445       |                        0.945478 |
| vanilla            | -        | 0.150659 |     0.652191 |       0.538324 |                      0.660341 |                        0.56703  |


### CovidIF

|                    |        ais |       amg |   single box |   single point |   iterative_prompts_start_box |   iterative_prompts_start_point |
|:-------------------|-----------:|----------:|-------------:|---------------:|------------------------------:|--------------------------------:|
| generalist         |   0.152222 | 0.0746765 |     0.569341 |      0.156363  |                      0.81381  |                       0.721318  |
| specialist_full_ft |   0.244846 | 0.13918   |     0.619487 |      0.244443  |                      0.825262 |                       0.749033  |
| specialist_lora    |   0.196347 | 0.0357408 |     0.583991 |      0.180315  |                      0.817916 |                       0.725224  |
| vanilla            | - | 0.0529358 |     0.388909 |      0.0424461 |                      0.29122  |                       0.0958208 |



# GoNuclear 

|                    |        ais |       amg |   single box |   single point |   iterative_prompts_start_box |   iterative_prompts_start_point |
|:-------------------|-----------:|----------:|-------------:|---------------:|------------------------------:|--------------------------------:|
| generalist         |    0.0890959 | 0.033125|    0.686063 |      0.167951|                      0.764031  |                       0.686301  |
| specialist_full_ft |   0.105786 | 0.030567   |    0.719985 |      0.244551  |                      0.744716 |                       0.653932  |
| specialist_lora    |   0.204244 |  0.043957 |     0.673355 |     0.268546 |                      0.756035 |                       0.684994  |
| vanilla            | - | -|     0.498223|      0.362247|                      0.439616|                       0.318307 |


## Extended Rank Study on Mitolab Glycolytic Muscle
### Mitolab Glycolytic Muscle

Domain is well known by the generalist model

#### From Default
| name    |      AIS |      AMG |      Box |    Point |
|:--------|---------:|---------:|---------:|---------:|
| vanilla | nan        | 0.150659 | 0.652191 | 0.538324 |
| lora_1  | 0.549484 | 0.507444 | 0.841889 | 0.711415 |
| lora_2  | 0.556883 | 0.490352 | 0.845439 | 0.706573 |
| lora_4  | 0.591906 | 0.489366 | 0.848887 | 0.702615 |
| lora_8  | 0.600346 | 0.476347 | 0.848355 | 0.693523 |
| lora_16 | 0.583292 | 0.471583 | 0.848447 | 0.709279 |
| lora_32 | 0.577014 | 0.505774 | 0.847235 | 0.696355 |
| lora_64 | 0.594854 | 0.531694 | 0.847471 | 0.733067 |
| full_ft | 0.654125 | 0.58816  | 0.844841 | 0.670931 |

#### From Generalist

| name    |      AIS |      AMG |      Box |    Point |
|:--------|---------:|---------:|---------:|---------:|
| generalist | 0.430989 | 0.357421 | 0.791705 | 0.519416 |
| lora_1  | 0.617624 | 0.539184 | 0.86804  | 0.727741 |
| lora_2  | 0.636927 | 0.531309 | 0.866054 | 0.72443  |
| lora_4  | 0.612037 | 0.555217 | 0.865789 | 0.722931 |
| lora_8  | 0.648352 | 0.564039 | 0.863956 | 0.730869 |
| lora_16 | 0.626368 | 0.559474 | 0.865566 | 0.727721 |
| lora_32 | 0.630273 | 0.562193 | 0.865755 | 0.72731  |
| lora_64 | 0.615694 | 0.572311 | 0.865311 | 0.742754 |
| full_ft | 0.681277 | 0.58494  | 0.854074 | 0.688395 |

### Orgasegment

Domain is highly different from domain ‘known’ by the generalist model

#### From Default
| name    |      AIS |      AMG |      Box |    Point |
|:--------|---------:|---------:|---------:|---------:|
| vanilla | nan        | 0.343003 | 0.630294 | 0.494892 |
| lora_1  | 0.414372 | 0.370126 | 0.767285 | 0.531497 |
| lora_2  | 0.426449 | 0.373112 | 0.770356 | 0.540227 |
| lora_4  | 0.429695 | 0.372823 | 0.76679  | 0.536833 |
| lora_8  | 0.418091 | 0.373189 | 0.769579 | 0.551802 |
| lora_16 | 0.419071 | 0.377984 | 0.773225 | 0.559315 |
| lora_32 | 0.421673 | 0.392801 | 0.77536  | 0.56451  |
| lora_64 | 0.431376 | 0.392141 | 0.77401  | 0.581975 |
| full_ft | 0.481982 | 0.450553 | 0.753868 | 0.568104 |


#### From Generalist

| name    |      AIS |      AMG |      Box |    Point |
|:--------|---------:|---------:|---------:|---------:|
| generalist | 0.239504 | 0.313557 | 0.766814 | 0.426106 |
| lora_1  | 0.467328 | 0.384649 | 0.788527 | 0.526489 |
| lora_2  | 0.471765 | 0.39285  | 0.792983 | 0.536056 |
| lora_4  | 0.455499 | 0.393884 | 0.796325 | 0.536955 |
| lora_8  | 0.451356 | 0.394755 | 0.795676 | 0.537157 |
| lora_16 | 0.476082 | 0.409496 | 0.787012 | 0.534475 |
| lora_32 | 0.471721 | 0.402015 | 0.794801 | 0.550029 |
| lora_64 | 0.474701 | 0.400378 | 0.798804 | 0.555428 |
| full_ft    | 0.504257 | 0.428629 | 0.77399  | 0.56243  |