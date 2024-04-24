# MAO-CLIMB: Curriculum Learning for Infant-inspired Model Building Beyond English

## Set-up 


```
git clone https://github.com/suchirsalhan/MAO-CLIMB
python3 -m venv venvs/demo; source venvs/demo/bin/activate
bash setup.sh
```
This will require being a member of the BabyLM HuggingFace and W&B accounts to provide the correct authorisation keys to log runs. 


## Training

Training logs are stored using Weights & Biases (W&B). This requires two parameters `experiment.group` and `experiment.name` to log runs. 

To train an SSLM for  `fr, de, ja, zh ` run the following command: 
```
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_strict_gold" tokenizer="zh_cbt"
```

For Dry Runs: 

```
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_strict_gold" tokenizer="zh_cbt" experiment.dry_run=True trainer.max_training_steps=100 trainer.num_warmup_steps=10

```

[Cambridge University HPC Cluster]: The models can be trained using the `wilkes3-gpu` on the Cambridge HPC cluster. Sample HPC scripts are provided in `./scripts`. 


To train an SSLM using the HPC, `cd scripts`, and then run the following command in the terminal: 
```
sbatch launch_slurm.wilkes3  experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_small" tokenizer="zh_cbt"
sbatch launch_slurm.wilkes3  experiment.name="german-demo-1" experiment.group="suchir-demo" dataset.subconfig="de_lang_small" tokenizer="de_cbt"
sbatch launch_slurm.wilkes3  experiment.name="french-demo-1" experiment.group="suchir-demo" dataset.subconfig="fr_lang_small" tokenizer="fr_cbt"

```

Example usage on the Interactive Node of the HPC for a Dry Run:

```
cd scripts
./launch_interactive.sh
cd ..
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_strict_gold" tokenizer="zh_cbt" experiment.dry_run=True trainer.max_training_steps=100 trainer.num_warmup_steps=10
```



## Training Datasets

HuggingFace BabyLM Datasets for French, German, Japanese and Chinese have been developed and released here:

[BabyLM](https://huggingface.co/datasets/cambridge-climb/BabyLM)


| Language  | code  | AO Corpora | Data Curriculum | Tokeniser |
| ------------- | ------------- | ------------- | ------------- |------------- |
| **French** |  `fr`  | `fr_lang_small` | `fr_lang_strict`  |`fr_cbt`  |
| **German** | `de`  | `de_lang_small`  | `de_lang_strict`  |`de_cbt`  |
| **Chinese** | `zh`  | `zh_lang_small` | `zh_lang_strict`  |`zh_cbt`  |
| **Spanish** | `es`  | `es_lang_small` | `es_lang_strict`  |`es_cbt`  |
| **Portuguese** | `po`  | `po_lang_small` | `po_lang_strict`  |`po_cbt`  |
| **Italian** | `it`  | `it_lang_small` | `it_lang_strict`  |`it_cbt`  |
| **Catalan** | `cat`  | `cat_lang_small` | `cat_lang_strict`  |`cat_cbt`  |
| **Dutch** | `nld`  | `nld_lang_small` | `nld_lang_strict`  |`nld_cbt`  |
| **Romanian** | `ron`  | `ron_lang_small` | `ron_lang_strict`  |`ron_cbt`  |
| **Russian** | `ru`  | `ru_lang_small` | `ru_lang_strict`  |`ru_cbt`  |
| **Polish** | `pol`  | `pol_lang_small` | `pol_lang_strict`  |`pol_cbt`  |
| **Bulgarian** | `bul`  | `bul_lang_small` | `bul_lang_strict`  |`bul_cbt`  |
| **Czech** | `ces`  | `ces_lang_small`  | `ces_lang_strict`  |`ces_cbt`  |
| **Swedish** | `swe`  | `swe_lang_small`  | `swe_lang_strict`  |`swe_cbt`  |
| **Norwegian** | `nor`  | `nor_lang_small`  | `nor_lang_strict`  |`nor_cbt`  |
| **Danish** | `dan`  | `dan_lang_small`  | `dan_lang_strict`  |`dan_cbt`  |
| **Iceland** | `isl`  | `isl_lang_small`  | `isl_lang_strict`  |`isl_cbt`  |
| **Korean** | `kor`  | `kor_lang_small`  | `kor_lang_strict`  |`kor_cbt`  |
| **Indonesian** | `ind`  | `ind_lang_small`  | `ind_lang_strict`  |`ind_cbt`  |
| **Thai** | `tha`  | `tha_lang_small`  | `tha_lang_strict`  |`tha_cbt`  |
| **Japanese** | `ja`  | `ja_lang_small`  | `ja_lang_strict`  |`ja_cbt`  |


Additionally, MAO-CHILDES corpora have been developed for low(er)-resourced languages:

| Language  | code  | AO Corpora | Data Curriculum | Tokeniser |
| ------------- | ------------- | ------------- | ------------- |------------- |
| **Afrikaans** |  `afr`  | `afr_lang_small` | `afr_lang_strict`  |`afr_cbt`  |
| **Croatian** | `hrv`  | `hrv_lang_small`  | `hrv_lang_strict`  |`hrv_cbt`  |
| **Serbian** | `srp`  | `srp_lang_small`  | `srp_lang_strict`  |`srp_cbt`  |
| **Slovenian** | `slv`  | `slv_lang_small`  | `slv_lang_strict`  |`slv_cbt`  |



## Evaluation



## 🧗 CLIMB 
The code is based on the Cambridge University & Collaborator's submission to the [Baby LM Challenge](https://babylm.github.io/) (strict-small track) for **English-based Small-Scale Language Models**. 

## Acknowledgements

Martinez, R. D., McGovern, H., Goriely, Z., Davis, C., Caines, A., Buttery, P., & Beinborn, L. (2023, December). CLIMB–Curriculum Learning for Infant-inspired Model Building. In Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning (pp. 112-127).


```
@inproceedings{martinez-etal-2023-climb,
    title = "{CLIMB} {--} Curriculum Learning for Infant-inspired Model Building",
    author = "Martinez, Richard Diehl  and
      McGovern, Hope  and
      Goriely, Zebulon  and
      Davis, Christopher  and
      Caines, Andrew  and
      Buttery, Paula  and
      Beinborn, Lisa",
    editor = "Warstadt, Alex  and
      Mueller, Aaron  and
      Choshen, Leshem  and
      Wilcox, Ethan  and
      Zhuang, Chengxu  and
      Ciro, Juan  and
      Mosquera, Rafael  and
      Paranjabe, Bhargavi  and
      Williams, Adina  and
      Linzen, Tal  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.conll-babylm.10",
    doi = "10.18653/v1/2023.conll-babylm.10",
    pages = "112--127",
}```
