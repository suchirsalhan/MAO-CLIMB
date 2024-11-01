# MAO-CLIMB: Curriculum Learning for Infant-inspired Model Building Beyond English

Cognitively-Plausible Small-Scale Language Models trained using developmentally-plausible corpora of Child-Directed Speech, and a series of universal and language-specific objective curricula.


## Set-up 


```
git clone https://github.com/suchirsalhan/MAO-CLIMB
python3 -m venv venvs/demo; source venvs/demo/bin/activate
bash setup.sh
```
This will require being a member of the BabyLM HuggingFace and W&B accounts to provide the correct authorisation keys to log runs. Save HuggingFace Read and Write Tokens as follows in `.env`: 
```
export HF_READ_TOKEN= [insert]
export HF_WRITE_TOKEN= [insert]

```

## Training

Training logs are stored using Weights & Biases (W&B). This requires two parameters `experiment.group` and `experiment.name` to log runs. 

To train an SSLM for  `fr, de, ja, zh ` run the following command: 
```
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_small" tokenizer="zh_cbt"
```

**IMPORTANT:** For `cat, ron`, the vocabulary sizes are slightly smaller. Specify this as follows: 
```
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_small" tokenizer="zh_cbt"
```

For Dry Runs: 

```
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_small" tokenizer="zh_cbt" experiment.dry_run=True trainer.max_training_steps=100 trainer.num_warmup_steps=10

```

## Bubbles


To train an SSLM using the HPC, `cd scripts`, and then run the following command in the terminal: 
```
sh launch_torchrun.sh experiment.name="chinese-demo-1" experiment.group="suchir-demo" ...
```

Note the following changes to the `setup.sh`. 

```
# module rm rhel7/global
# module rm rhel7/default-gpu

if [ ! -d "env" ]; then
        # module load python-3.9.6-gcc-5.4.0-sbr552h
        export TMPDIR='/var/tmp'
        virtualenv -p python3 env
        source env/bin/activate
        git lfs install
        pip install -r requirements.txt
        pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
        pre-commit install
        huggingface-cli login
        wandb login
else
        source env/bin/activate
fi
source .env


export PATH="$(pwd)/lib/bin:$PATH"
```

## HPC

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
The code extends Cambridge University & Collaborator's submission to the [Baby LM Challenge](https://babylm.github.io/) (strict-small track) for **English-based Small-Scale Language Models**. 

## Citation

If you find the code or ideas behind the paper useful, please consider citing our paper. 

Salhan, S.A., Martinez, R. D.,  Goriely, Z., & Buttery, P. (2024, November). Less is More: Pre-Training Cross-Lingual Small-Scale Language Models with Cognitively-Plausible Curriculum Learning Strategies. In Proceedings of the BabyLM Challenge at the 28th Conference on Computational Natural Language Learning (pp. 112-127).


```
@inproceedings{salhan-etal-2024-less,
    title = " Less is More: Pre-Training Cross-Lingual Small-Scale Language Models with Cognitively-Plausible Curriculum Learning Strategies",
    author = "Salhan, Suchir  and
      Diehl Martinez, Richard
      Goriely, Zebulon  and
      Buttery, Paula",
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
    booktitle = "Proceedings of the BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.conll-babylm.10",
    doi = "10.18653/v1/2023.conll-babylm.10",
    pages = "112--127",
}```

