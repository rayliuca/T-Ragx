| run                                | sacrebleu   | chrf        | meteor     | comet22    | normed_mean   | standard_mean   |
|:-----------------------------------|:------------|:------------|:-----------|:-----------|:--------------|:----------------|
| seamless                           | 5.2327      | 22.3017     | 0.2158     | 0.3334     | 0.5803        | -3.3062         |
| original_mistral                   | 8.8418      | 37.3060     | 0.3575     | 0.3832     | 0.8579        | -0.7707         |
| original_alma_7b_r_rag             | 9.5909      | 35.5959     | 0.3341     | **0.4055** | 0.8608        | -0.4923         |
| original_mistral_rag               | 8.4550      | 37.6006     | 0.3754     | 0.3817     | 0.8625        | -0.7392         |
| original_mistral_rag_pretext       | 9.3102      | 38.0327     | 0.3825     | 0.3834     | 0.8861        | -0.5990         |
| original_alma_7b_r_rag_pretext     | 10.3277     | 36.4520     | 0.3538     | 0.3985     | 0.8869        | -0.4491         |
| original_alma_7b_r                 | 11.1073     | 38.1487     | 0.3677     | 0.3868     | 0.9126        | -0.4669         |
| offical_tower                      | 14.9021     | 40.6960     | 0.4186     | 0.3924     | 1.0315        | 0.2278          |
| qlora_internlm2_rag                | 15.0913     | 40.3777     | 0.4293     | 0.3918     | 1.0390        | 0.2546          |
| qlora_mistral                      | 15.4810     | 40.0728     | 0.4294     | 0.3904     | 1.0432        | 0.2376          |
| qlora_NeuralOmniBeagle             | 15.7247     | 40.5043     | 0.4390     | 0.3910     | 1.0566        | 0.3274          |
| qlora_mistral_rag                  | 15.6427     | 41.3578     | 0.4527     | 0.3909     | 1.0686        | 0.4191          |
| madlad400_10b                      | 16.1251     | 41.7136     | 0.4420     | 0.3922     | 1.0740        | 0.4539          |
| qlora_tower_rag                    | 16.0539     | 41.7045     | 0.4513     | 0.3910     | 1.0775        | 0.4617          |
| qlora_tower                        | 16.3245     | 41.6093     | 0.4476     | 0.3922     | 1.0803        | 0.4823          |
| qlora_NeuralOmniBeagle_rag         | 16.2081     | 41.8936     | 0.4607     | 0.3904     | 1.0867        | 0.5087          |
| qlora_internlm2                    | **17.4556** | 42.3955     | 0.4544     | 0.3912     | 1.1092        | 0.6093          |
| qlora_internlm2_rag_pretext        | 16.7694     | 43.1506     | 0.4693     | 0.3907     | 1.1102        | 0.6536          |
| qlora_NeuralOmniBeagle_rag_pretext | 17.3697     | 43.2671     | 0.4729     | 0.3902     | 1.1236        | 0.7051          |
| qlora_tower_rag_pretext            | 17.2946     | **43.3735** | 0.4764     | 0.3917     | 1.1260        | **0.7472**      |
| qlora_mistral_rag_pretext          | 17.3077     | 43.3310     | **0.4782** | 0.3907     | **1.1265**    | 0.7352          |