# MAO-CLIMB: Curriculum Learning for Infant-inspired Model Building Beyond English

## Set-up 
HuggingFace BabyLM Datasets for French, German, Japanese and Chinese have been developed and released here:

```
git clone https://github.com/suchirsalhan/MAO-CLIMB
python3 -m venv venvs/demo; source venvs/demo/bin/activate
bash setup.sh
```

[BabyLM](https://huggingface.co/datasets/cambridge-climb/BabyLM)

Training logs are stored using Weights & Biases (W&B). This requires two parameters `experiment.group` and `experiment.name` to log runs. 

To train an SSLM for  `fr, de, ja, zh ` run the following command: 
```
python train.py experiment.name="japanese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_strict_gold" tokenizer.subconfig="zh_cbt"
```


For Dry Runs: 

```
python train.py experiment.name="chinese-demo-1" experiment.group="suchir-demo" dataset.subconfig="zh_lang_strict_gold" experiment.dry_run=True trainer.max_training_steps=100 trainer.num_warmup_steps=10

```

[Cambridge University HPC Cluster]: The models can be trained using the `wilkes3-gpu` on the Cambridge HPC cluster. Sample HPC scripts are provided in `./scripts`. 


To train an SSLM using the HPC, `cd scripts`, and then run the following command in the terminal: 
```
sbatch launch_slurm.wilkes3 experiment.name="japanese-demo-1" experiment.group="suchir-demo" dataset.subconfig="ja_lang_strict_gold"
```

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
