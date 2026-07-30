# Hallucination Pattern Detection — results summary

## Dataset: fever

| model       | method                    |   layer |    auroc |   accuracy |         f1 |
|:------------|:--------------------------|--------:|---------:|-----------:|-----------:|
| llama3.1-8b | INSIDE                    |      -1 | 0.518124 |   0.5375   | 0.574713   |
| llama3.1-8b | attention_entropy         |      16 | 0.686279 |   0.6825   | 0.653951   |
| llama3.1-8b | probe_linear              |      17 | 0.954763 |   0.858333 | 0.847321   |
| llama3.1-8b | probe_mlp                 |      17 | 0.953304 |   0.8625   | 0.869984   |
| llama3.1-8b | self_consistency_exact    |      -1 | 0.5      |   0.5125   | 0.677686   |
| llama3.1-8b | self_consistency_semantic |      -1 | 0.467567 |   0.51     | 0.581197   |
| mistral-7b  | INSIDE                    |      -1 | 0.526779 |   0.56     | 0.553299   |
| mistral-7b  | attention_entropy         |      16 | 0.580063 |   0.5775   | 0.662675   |
| mistral-7b  | probe_linear              |      14 | 0.959767 |   0.9      | 0.903766   |
| mistral-7b  | probe_mlp                 |      15 | 0.95935  |   0.9      | 0.904041   |
| mistral-7b  | self_consistency_exact    |      -1 | 0.5      |   0.5125   | 0.677686   |
| mistral-7b  | self_consistency_semantic |      -1 | 0.42469  |   0.5025   | 0.167364   |
| qwen2.5-7b  | INSIDE                    |      -1 | 0.496961 |   0.52     | 0.319149   |
| qwen2.5-7b  | attention_entropy         |      27 | 0.608931 |   0.6125   | 0.498382   |
| qwen2.5-7b  | probe_linear              |      20 | 0.920575 |   0.820833 | 0.810653   |
| qwen2.5-7b  | probe_mlp                 |      21 | 0.927038 |   0.854167 | 0.863635   |
| qwen2.5-7b  | self_consistency_exact    |      -1 | 0.502439 |   0.49     | 0.00970874 |
| qwen2.5-7b  | self_consistency_semantic |      -1 | 0.465666 |   0.5275   | 0.676923   |

## Dataset: halueval_qa

| model       | method                    |   layer |    auroc |   accuracy |         f1 |
|:------------|:--------------------------|--------:|---------:|-----------:|-----------:|
| llama3.1-8b | INSIDE                    |      -1 | 0.528938 |   0.5525   | 0.588506   |
| llama3.1-8b | attention_entropy         |       0 | 0.940924 |   0.8625   | 0.857881   |
| llama3.1-8b | probe_linear              |      15 | 0.998125 |   0.979167 | 0.979625   |
| llama3.1-8b | probe_mlp                 |      15 | 0.997708 |   0.970833 | 0.971495   |
| llama3.1-8b | self_consistency_exact    |      -1 | 0.502288 |   0.5      | 0.0740741  |
| llama3.1-8b | self_consistency_semantic |      -1 | 0.488512 |   0.52     | 0.625      |
| mistral-7b  | INSIDE                    |      -1 | 0.438536 |   0.5025   | 0.66778    |
| mistral-7b  | attention_entropy         |       0 | 0.902298 |   0.85     | 0.840426   |
| mistral-7b  | probe_linear              |      18 | 0.998333 |   0.966667 | 0.96748    |
| mistral-7b  | probe_mlp                 |      18 | 0.998125 |   0.970833 | 0.971495   |
| mistral-7b  | self_consistency_exact    |      -1 | 0.466524 |   0.5      | 0.00990099 |
| mistral-7b  | self_consistency_semantic |      -1 | 0.464612 |   0.505    | 0.123894   |
| qwen2.5-7b  | INSIDE                    |      -1 | 0.490887 |   0.515    | 0.378205   |
| qwen2.5-7b  | attention_entropy         |       0 | 0.866147 |   0.8025   | 0.805897   |
| qwen2.5-7b  | probe_linear              |      19 | 0.996875 |   0.970833 | 0.971191   |
| qwen2.5-7b  | probe_mlp                 |      25 | 0.9975   |   0.966667 | 0.96748    |
| qwen2.5-7b  | self_consistency_exact    |      -1 | 0.505063 |   0.505    | 0.038835   |
| qwen2.5-7b  | self_consistency_semantic |      -1 | 0.521663 |   0.54     | 0.665455   |

## Dataset: synthetic

| model       | method                    |   layer |    auroc |   accuracy |       f1 |
|:------------|:--------------------------|--------:|---------:|-----------:|---------:|
| llama3.1-8b | INSIDE                    |      -1 | 0.518475 |   0.54     | 0.323529 |
| llama3.1-8b | attention_entropy         |      31 | 0.6393   |   0.6375   | 0.640199 |
| llama3.1-8b | probe_linear              |      32 | 1        |   0.933333 | 0.937928 |
| llama3.1-8b | probe_mlp                 |      28 | 1        |   0.9875   | 0.987755 |
| llama3.1-8b | self_consistency_exact    |      -1 | 0.5      |   0.5      | 0.666667 |
| llama3.1-8b | self_consistency_semantic |      -1 | 0.50245  |   0.525    | 0.379085 |
| mistral-7b  | INSIDE                    |      -1 | 0.528487 |   0.54     | 0.523316 |
| mistral-7b  | attention_entropy         |      16 | 0.7737   |   0.735    | 0.725389 |
| mistral-7b  | probe_linear              |      18 | 0.998333 |   0.9125   | 0.921572 |
| mistral-7b  | probe_mlp                 |      20 | 0.99875  |   0.975    | 0.974326 |
| mistral-7b  | self_consistency_exact    |      -1 | 0.5      |   0.5      | 0.666667 |
| mistral-7b  | self_consistency_semantic |      -1 | 0.50245  |   0.5275   | 0.403785 |
| qwen2.5-7b  | INSIDE                    |      -1 | 0.488825 |   0.5175   | 0.498701 |
| qwen2.5-7b  | attention_entropy         |      27 | 0.6828   |   0.6625   | 0.536082 |
| qwen2.5-7b  | probe_linear              |      17 | 1        |   0.9625   | 0.964229 |
| qwen2.5-7b  | probe_mlp                 |      18 | 1        |   0.991667 | 0.99177  |
| qwen2.5-7b  | self_consistency_exact    |      -1 | 0.5      |   0.5      | 0.666667 |
| qwen2.5-7b  | self_consistency_semantic |      -1 | 0.5018   |   0.5225   | 0.56492  |

## Dataset: truthfulqa

| model       | method                    |   layer |    auroc |   accuracy |        f1 |
|:------------|:--------------------------|--------:|---------:|-----------:|----------:|
| llama3.1-8b | INSIDE                    |      -1 | 0.432523 |   0.5275   | 0.681282  |
| llama3.1-8b | attention_entropy         |      31 | 0.603066 |   0.5975   | 0.570667  |
| llama3.1-8b | probe_linear              |      13 | 0.913696 |   0.825    | 0.832983  |
| llama3.1-8b | probe_mlp                 |      14 | 0.913696 |   0.8125   | 0.818498  |
| llama3.1-8b | self_consistency_exact    |      -1 | 0.5      |   0.51     | 0.675497  |
| llama3.1-8b | self_consistency_semantic |      -1 | 0.510729 |   0.5325   | 0.490463  |
| mistral-7b  | INSIDE                    |      -1 | 0.52536  |   0.5575   | 0.645291  |
| mistral-7b  | attention_entropy         |      31 | 0.581032 |   0.59     | 0.531429  |
| mistral-7b  | probe_linear              |      16 | 0.903898 |   0.8      | 0.806514  |
| mistral-7b  | probe_mlp                 |      16 | 0.908068 |   0.820833 | 0.829181  |
| mistral-7b  | self_consistency_exact    |      -1 | 0.502138 |   0.4975   | 0.0382775 |
| mistral-7b  | self_consistency_semantic |      -1 | 0.541141 |   0.5525   | 0.646943  |
| qwen2.5-7b  | INSIDE                    |      -1 | 0.505927 |   0.54     | 0.621399  |
| qwen2.5-7b  | attention_entropy         |      14 | 0.48857  |   0.5175   | 0.498701  |
| qwen2.5-7b  | probe_linear              |      19 | 0.91453  |   0.808333 | 0.819275  |
| qwen2.5-7b  | probe_mlp                 |      19 | 0.924953 |   0.833333 | 0.841674  |
| qwen2.5-7b  | self_consistency_exact    |      -1 | 0.502076 |   0.4975   | 0.0382775 |
| qwen2.5-7b  | self_consistency_semantic |      -1 | 0.478066 |   0.515    | 0.605691  |
