---
base_model: DeepPavlov/rubert-base-cased-sentence
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4720
- loss:CosineSimilarityLoss
widget:
- source_sentence: 'Под другим углом | Выпуск 3 | Астрология Давайте погадаем, на
    что же в новом выпуске "Под другим углом" мы посмотрим. На астро… -номию?! Нет!
    Астро… -физику?! Нееее! Цветы астры?! Да нет! Астрологию! Мы разберёмся в её принципах.
    Поймём, какое созвездие из зодиакального круга потерялось. Что такое ментальные
    карты? Выясним есть ли тут наука или нет. Как связаны астрология и психология?
    И примем решение, верить ли гороскопам (Спойлер: это решение повлияет на вашу
    судьбу).'
  sentences:
  - 'Путешествия: Тип путешествия: Бюджетный отдых'
  - 'Изобразительное искусство: Современное искусство'
  - 'События и достопримечательности: Бизнес-выставки и конференции'
- source_sentence: Артмеханика. Интервью. Денис Рогов. Что такое метавселенная? Где
    мы можем ощутить как она работает уже сегодня? Кто создает альтернативные реальности?
  sentences:
  - 'Музыка и аудио: Рок музыка: Легкий рок'
  - Карьера
  - 'Образование: Обучение детей с особыми потребностями'
- source_sentence: Артмеханика. Концерт группы Диктофон. Концерт группы Диктофон.
  sentences:
  - 'События и достопримечательности: Спортивные события'
  - 'Массовая культура: Юмор и сатира'
  - 'События и достопримечательности: Концерты и музыкальные мероприятия'
- source_sentence: Один выходной | Выпуск 21 | Спорт Сегодня Ташэ займется спортом
    и расскажет вам, где можно не просто провести свой выходной, а еще и покачаться
    к летнему сезону, как всегда за деньги и бесплатно.
  sentences:
  - 'События и достопримечательности: Парки развлечений и тематические парки'
  - 'Бизнес и финансы: Промышленность и сфера услуг: Швейная промышленность'
  - 'Массовая культура: Смерти знаменитостей'
- source_sentence: «МузЛофт - Подкаст» с Мартиросяном, Сорокиным, Авериным, Матуа
    - 3 июня в 15:00! Подписывайся, чтобы не пропустить!
  sentences:
  - 'Медицина: Медицинские направления: Эндокринология'
  - 'Массовая культура: Юмор и сатира'
  - 'События и достопримечательности: Концерты и музыкальные мероприятия'
---

# SentenceTransformer based on DeepPavlov/rubert-base-cased-sentence

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [DeepPavlov/rubert-base-cased-sentence](https://huggingface.co/DeepPavlov/rubert-base-cased-sentence). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [DeepPavlov/rubert-base-cased-sentence](https://huggingface.co/DeepPavlov/rubert-base-cased-sentence) <!-- at revision 78b5122d6365337dd4114281b0d08cd1edbb3bc8 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '«МузЛофт - Подкаст» с Мартиросяном, Сорокиным, Авериным, Матуа - 3 июня в 15:00! Подписывайся, чтобы не пропустить!',
    'События и достопримечательности: Концерты и музыкальные мероприятия',
    'Медицина: Медицинские направления: Эндокринология',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 4,720 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                       | label                                                            |
  |:--------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------|
  | type    | string                                                                              | string                                                                           | float                                                            |
  | details | <ul><li>min: 14 tokens</li><li>mean: 89.78 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 2 tokens</li><li>mean: 8.85 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 0.05</li><li>mean: 0.38</li><li>max: 0.99</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_1                                                    | label             |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|:------------------|
  | <code>МАКСИМ НАРОДНЫЙ Выпуск №13 ГОТОВИМ ЧАШУШУЛИ ПО-ГРУЗИНСКИ Предлагаю подписчикам приготовить чашушули по-грузински.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>Бизнес и финансы: Экономика: Цены на бензин</code>      | <code>0.1</code>  |
  | <code>«МузЛофт-Подкаст» с Сарухановым — 20 ноября в 15:00! Подписывайся, чтобы не пропустить!</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>Массовая культура</code>                                | <code>0.85</code> |
  | <code>Команда "3/21" в ГрандТуре-2022: село Тагархай Команда "3/21" приехала в село Тагархай, на родину Героя Советского Союза, легендарного бурятского снайпера Шамбыла Ешеевича Тулаева. Из маленького бурятского села на Великую Отечественную войну ушли 120 человек. 55 тагархайцев вернулись после войны к родным.  Аккаунт команды в Rutube - / ………… ГрандТур «Байкальская миля 2022. Места силы» - это грандиозное автопутешествие, состоявшееся с 20 февраля по 9 марта. В этом пробеге приняли участие 14 команд, стартовавших из Санкт-Петербурга, Москвы, Казани и Владивостока и финишировавших в селе Максимиха, месте проведения Фестиваля скорости «Байкальская миля». Каждая команда получила свой уникальный маршрут и по 6 заданий, разделенных на темы: «География», «Патриотика», «Экология», «Общество», «Моя «Байкальская миля» и «Самостоятельное задание», в рамках которого участники могли рассказать своим зрителям о том, что считают нужным и важным. #baikalmile2022 #ГрандТурБайкальскаяМиля2022 #местасилы #энтузиастыскорости #гкфск #байкальскаямиля #baikalmile #водоходъ #rutube #мотовесна #finecustommechanics #fire_center #РГО #бурятиябольшечембайкал #МузейПобеды #ркспорт</code> | <code>События и достопримечательности: Торговые центры</code> | <code>0.15</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.6949 | 500  | 0.1104        |
| 3.3898 | 1000 | 0.0502        |
| 5.0847 | 1500 | 0.0313        |
| 6.7797 | 2000 | 0.0234        |
| 8.4746 | 2500 | 0.0168        |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.1.1
- Transformers: 4.44.2
- PyTorch: 2.4.1+cu121
- Accelerate: 0.34.2
- Datasets: 3.0.1
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->