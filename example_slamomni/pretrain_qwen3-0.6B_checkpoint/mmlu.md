# MMLU

Evaluated by `lm-evaluation-harness`.

```bash
singularity exec --nv \
  --env SSL_CERT_FILE=/workspace/cacert.pem \
  -B /work/u3937558/slamkit:/workspace \
  -B /work/u3937558/.cache:/tmp/cache \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "export HF_HOME=/tmp/cache/huggingface && \
    export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
    export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
    cd /workspace/lm-evaluation-harness && source venv/bin/activate && \
    lm_eval --model hf \
      --model_args pretrained=Qwen/Qwen3-0.6B \
      --tasks mmlu \
      --device cuda:0 \
      --batch_size 8"
```

|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.4034|±  |0.0040|
| - humanities                          |      2|none  |      |acc   |↑  |0.3671|±  |0.0069|
|  - formal_logic                       |      1|none  |     0|acc   |↑  |0.4048|±  |0.0439|
|  - high_school_european_history       |      1|none  |     0|acc   |↑  |0.5576|±  |0.0388|
|  - high_school_us_history             |      1|none  |     0|acc   |↑  |0.5098|±  |0.0351|
|  - high_school_world_history          |      1|none  |     0|acc   |↑  |0.5823|±  |0.0321|
|  - international_law                  |      1|none  |     0|acc   |↑  |0.5620|±  |0.0453|
|  - jurisprudence                      |      1|none  |     0|acc   |↑  |0.4259|±  |0.0478|
|  - logical_fallacies                  |      1|none  |     0|acc   |↑  |0.4663|±  |0.0392|
|  - moral_disputes                     |      1|none  |     0|acc   |↑  |0.3266|±  |0.0252|
|  - moral_scenarios                    |      1|none  |     0|acc   |↑  |0.2413|±  |0.0143|
|  - philosophy                         |      1|none  |     0|acc   |↑  |0.4148|±  |0.0280|
|  - prehistory                         |      1|none  |     0|acc   |↑  |0.4383|±  |0.0276|
|  - professional_law                   |      1|none  |     0|acc   |↑  |0.3005|±  |0.0117|
|  - world_religions                    |      1|none  |     0|acc   |↑  |0.5322|±  |0.0383|
| - other                               |      2|none  |      |acc   |↑  |0.4271|±  |0.0087|
|  - business_ethics                    |      1|none  |     0|acc   |↑  |0.4300|±  |0.0498|
|  - clinical_knowledge                 |      1|none  |     0|acc   |↑  |0.3358|±  |0.0291|
|  - college_medicine                   |      1|none  |     0|acc   |↑  |0.2948|±  |0.0348|
|  - global_facts                       |      1|none  |     0|acc   |↑  |0.2500|±  |0.0435|
|  - human_aging                        |      1|none  |     0|acc   |↑  |0.4709|±  |0.0335|
|  - management                         |      1|none  |     0|acc   |↑  |0.5631|±  |0.0491|
|  - marketing                          |      1|none  |     0|acc   |↑  |0.6453|±  |0.0313|
|  - medical_genetics                   |      1|none  |     0|acc   |↑  |0.3700|±  |0.0485|
|  - miscellaneous                      |      1|none  |     0|acc   |↑  |0.4917|±  |0.0179|
|  - nutrition                          |      1|none  |     0|acc   |↑  |0.4542|±  |0.0285|
|  - professional_accounting            |      1|none  |     0|acc   |↑  |0.2908|±  |0.0271|
|  - professional_medicine              |      1|none  |     0|acc   |↑  |0.3199|±  |0.0283|
|  - virology                           |      1|none  |     0|acc   |↑  |0.4518|±  |0.0387|
| - social sciences                     |      2|none  |      |acc   |↑  |0.4797|±  |0.0089|
|  - econometrics                       |      1|none  |     0|acc   |↑  |0.2982|±  |0.0430|
|  - high_school_geography              |      1|none  |     0|acc   |↑  |0.4596|±  |0.0355|
|  - high_school_government_and_politics|      1|none  |     0|acc   |↑  |0.5337|±  |0.0360|
|  - high_school_macroeconomics         |      1|none  |     0|acc   |↑  |0.4128|±  |0.0250|
|  - high_school_microeconomics         |      1|none  |     0|acc   |↑  |0.4118|±  |0.0320|
|  - high_school_psychology             |      1|none  |     0|acc   |↑  |0.5651|±  |0.0213|
|  - human_sexuality                    |      1|none  |     0|acc   |↑  |0.5038|±  |0.0439|
|  - professional_psychology            |      1|none  |     0|acc   |↑  |0.4101|±  |0.0199|
|  - public_relations                   |      1|none  |     0|acc   |↑  |0.4545|±  |0.0477|
|  - security_studies                   |      1|none  |     0|acc   |↑  |0.5143|±  |0.0320|
|  - sociology                          |      1|none  |     0|acc   |↑  |0.6468|±  |0.0338|
|  - us_foreign_policy                  |      1|none  |     0|acc   |↑  |0.5800|±  |0.0496|
| - stem                                |      2|none  |      |acc   |↑  |0.3597|±  |0.0084|
|  - abstract_algebra                   |      1|none  |     0|acc   |↑  |0.3000|±  |0.0461|
|  - anatomy                            |      1|none  |     0|acc   |↑  |0.3704|±  |0.0417|
|  - astronomy                          |      1|none  |     0|acc   |↑  |0.4737|±  |0.0406|
|  - college_biology                    |      1|none  |     0|acc   |↑  |0.4653|±  |0.0417|
|  - college_chemistry                  |      1|none  |     0|acc   |↑  |0.3300|±  |0.0473|
|  - college_computer_science           |      1|none  |     0|acc   |↑  |0.2500|±  |0.0435|
|  - college_mathematics                |      1|none  |     0|acc   |↑  |0.3200|±  |0.0469|
|  - college_physics                    |      1|none  |     0|acc   |↑  |0.2647|±  |0.0439|
|  - computer_security                  |      1|none  |     0|acc   |↑  |0.6200|±  |0.0488|
|  - conceptual_physics                 |      1|none  |     0|acc   |↑  |0.3872|±  |0.0318|
|  - electrical_engineering             |      1|none  |     0|acc   |↑  |0.4207|±  |0.0411|
|  - elementary_mathematics             |      1|none  |     0|acc   |↑  |0.3624|±  |0.0248|
|  - high_school_biology                |      1|none  |     0|acc   |↑  |0.4323|±  |0.0282|
|  - high_school_chemistry              |      1|none  |     0|acc   |↑  |0.3251|±  |0.0330|
|  - high_school_computer_science       |      1|none  |     0|acc   |↑  |0.4300|±  |0.0498|
|  - high_school_mathematics            |      1|none  |     0|acc   |↑  |0.2815|±  |0.0274|
|  - high_school_physics                |      1|none  |     0|acc   |↑  |0.2318|±  |0.0345|
|  - high_school_statistics             |      1|none  |     0|acc   |↑  |0.2315|±  |0.0288|
|  - machine_learning                   |      1|none  |     0|acc   |↑  |0.3839|±  |0.0462|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.4034|±  |0.0040|
| - humanities     |      2|none  |      |acc   |↑  |0.3671|±  |0.0069|
| - other          |      2|none  |      |acc   |↑  |0.4271|±  |0.0087|
| - social sciences|      2|none  |      |acc   |↑  |0.4797|±  |0.0089|
| - stem           |      2|none  |      |acc   |↑  |0.3597|±  |0.0084|