# Bảng hàm thuật toán — Cell & Dòng gọi

> Số **Cell** đếm từ 1 (cell code đầu tiên trong notebook, bỏ qua markdown).

> **Dòng** là số dòng trong source của cell đó.

---


## Phase 1 — EDA


### `01_EDA_Preprocessing.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `_normalize_name()` | **6**, dòng **66** | C**6** L**90**; C**6** L**99**; C**6** L**108**; C**6** L**109**; C**6** L**110** |
| `get_top_ngrams_by_label()` | **31**, dòng **6** | C**31** L**82**; C**31** L**83**; C**31** L**84**; C**31** L**85**; C**31** L**86** (+1) |
| `infer_column()` | **6**, dòng **71** | C**8** L**1**; C**8** L**2**; C**8** L**7**; C**8** L**8**; C**8** L**13** (+7) |
| `label_distribution_dict()` | **22**, dòng **2** | C**22** L**55**; C**22** L**56**; C**22** L**57**; C**22** L**58**; C**22** L**59** (+3) |
| `normalize_label_series()` | **14**, dòng **13** | C**15** L**31** |
| `parse_review_datetime()` | **33**, dòng **2** | C**33** L**27**; C**33** L**28**; `02_Feature_Engineering.ipynb` C**15** L**16**; `02_Feature_Engineering.ipynb` C**15** L**17**; `02_Feature_Engineering.ipynb` C**15** L**28** |
| `record_advanced_artifact()` | **25**, dòng **63** | C**27** L**43**; C**27** L**65**; C**27** L**66**; C**27** L**86**; C**27** L**87** (+46) |

## Phase 2 — ModernBERT & Behavioral


### `02_Feature_Engineering.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `add_advanced_behavioral_features()` | **16**, dòng **29** | C**16** L**77**; C**16** L**78** |
| `add_basic_behavioral_features()` | **14**, dòng **20** | C**14** L**70**; C**14** L**71** |
| `array_memory_mb()` | **20**, dòng **2** | C**20** L**44**; C**20** L**46**; `03_PCA_Feature_Selection.ipynb` C**8** L**48**; `03_PCA_Feature_Selection.ipynb` C**8** L**49**; `03_PCA_Feature_Selection.ipynb` C**16** L**53** |
| `extract_or_load_bert_embeddings()` | **10**, dòng **72** | C**11** L**16**; C**11** L**21**; C**11** L**22**; C**11** L**27**; C**11** L**28** |
| `first_existing_column()` | **7**, dòng **51** | C**7** L**67**; C**7** L**71**; C**7** L**73**; C**7** L**75**; C**7** L**77** (+2) |
| `fit_reviewer_behavior_map()` | **15**, dòng **142** | C**15** L**209** |
| `load_bert_model()` | **10**, dòng **32** | C**11** L**2** |
| `load_sentiment_scorer()` | **13**, dòng **26** | C**13** L**78** |
| `mean_pool_last_hidden_state()` | **10**, dòng **18** | C**10** L**129** |
| `parse_review_datetime()` | **13**, dòng **6** | C**15** L**16**; C**15** L**17**; C**15** L**28**; C**15** L**29**; C**15** L**88**; `01_EDA_Preprocessing.ipynb` C**33** L**27**; `01_EDA_Preprocessing.ipynb` C**33** L**28** (+5) |
| `previous_time_gap_hours_for_targets()` | **15**, dòng **74** | C**16** L**15**; C**16** L**62** |
| `prior_window_count_for_targets()` | **15**, dòng **2** | C**16** L**36**; C**16** L**37**; C**16** L**42**; C**16** L**43** |
| `schema_value()` | **7**, dòng **33** | C**7** L**67**; C**7** L**69**; C**7** L**71**; C**7** L**73**; C**7** L**75** (+1) |
| `valid_cached_embedding()` | **10**, dòng **54** | C**10** L**75**; C**10** L**76** |

## Phase 3 — PCA/SVD


### `03_PCA_Feature_Selection.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `array_memory_mb()` | **8**, dòng **2** | C**8** L**48**; C**8** L**49**; C**16** L**53**; C**16** L**54**; C**16** L**71**; `02_Feature_Engineering.ipynb` C**20** L**44**; `02_Feature_Engineering.ipynb` C**20** L**46**; `07_Evaluation_Ablation.ipynb` C**15** L**105** (+1) |
| `choose_component_count()` | **12**, dòng **78** | C**13** L**46** |
| `fit_final_reducer()` | **12**, dòng **52** | C**13** L**49** |
| `fit_selection_reducer()` | **12**, dòng **10** | C**13** L**19** |
| `is_memory_exception()` | **12**, dòng **2** | C**13** L**29**; C**13** L**30** |
| `load_float32()` | **10**, dòng **2** | C**10** L**16** |
| `process_memory_mb()` | **8**, dòng **18** | C**8** L**50**; C**8** L**51**; C**16** L**55**; C**16** L**56**; C**16** L**73** (+1) |
| `variance_at()` | **12**, dòng **85** | C**12** L**109**; C**12** L**111**; C**12** L**160**; C**12** L**161** |

## Phase 4 — PSO + DL


### `04_PSO_Model_Training.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__init__()` | **13**, dòng **22** | C**13** L**23**; C**13** L**24**; C**13** L**70**; C**13** L**71**; C**13** L**135**; `05_03_MLP_Raw.ipynb` C**7** L**5**; `05_03_MLP_Raw.ipynb` C**7** L**6**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**8** L**40** (+1) |
| `build_model()` | **13**, dòng **175** | C**14** L**168**; `06_Adversarial_XAI.ipynb` C**16** L**135** |
| `capped_batch_size()` | **6**, dòng **103** | C**9** L**28**; C**14** L**229**; C**14** L**230**; C**16** L**25**; C**18** L**167** (+5) |
| `compute_class_weights()` | **9**, dòng **60** | C**9** L**74** |
| `compute_pso_objective_score()` | **18**, dòng **176** | C**19** L**73** |
| `decode_particle()` | **18**, dòng **146** | C**19** L**10**; C**20** L**162** |
| `environment_versions()` | **11**, dòng **124** | C**23** L**89**; C**23** L**90**; `07_Evaluation_Ablation.ipynb` C**25** L**137**; `07_Evaluation_Ablation.ipynb` C**25** L**138** |
| `evaluate_model()` | **14**, dòng **106** | C**14** L**189**; C**14** L**190**; C**16** L**100**; C**16** L**101**; C**22** L**85** (+1) |
| `evaluate_predictions()` | **11**, dòng **32** | C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346**; `05_02_XGBoost_Raw.ipynb` C**5** L**346** |
| `expand_component_block_weights()` | **13**, dòng **2** | C**13** L**77**; `06_Adversarial_XAI.ipynb` C**16** L**51** |
| `fallback_pso()` | **20**, dòng **2** | C**20** L**102** |
| `fit_model()` | **14**, dòng **144** | C**16** L**33**; C**16** L**34**; C**19** L**44**; C**19** L**45**; C**22** L**18** (+1) |
| `forward()` | **13**, dòng **37** | *(method/callback — xem mục bổ sung)* |
| `load_split_arrays()` | **9**, dòng **2** | C**9** L**71**; C**9** L**72** |
| `make_loader()` | **9**, dòng **26** | C**9** L**77**; C**9** L**79**; C**9** L**81**; C**16** L**27**; C**16** L**29**; `05_03_MLP_Raw.ipynb` C**7** L**41**; `05_03_MLP_Raw.ipynb` C**7** L**42**; `05_03_MLP_Raw.ipynb` C**7** L**76** (+6) |
| `nearest_choice()` | **18**, dòng **140** | C**18** L**150**; C**18** L**157**; C**18** L**158**; C**18** L**161**; C**18** L**162** (+4) |
| `normalize_pso_result()` | **20**, dòng **49** | C**20** L**95**; C**20** L**96**; C**20** L**117**; C**20** L**118** |
| `plot_training_history()` | **11**, dòng **150** | C**16** L**114**; C**16** L**115**; C**22** L**125**; C**22** L**126** |
| `predict_probabilities()` | **14**, dòng **52** | C**14** L**119**; C**14** L**120** |
| `probe_npy()` | **8**, dòng **10** | C**8** L**54**; C**8** L**56** |
| `pso_objective()` | **19**, dòng **6** | *(method/callback — xem mục bổ sung)* |
| `read_json()` | **8**, dòng **2** | C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **11**, dòng **20** | C**11** L**99**; C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243**; `05_00_Phase5_Run_Order.ipynb` C**5** L**244**; `05_01_LightGBM_Raw.ipynb` C**5** L**313** |
| `safe_roc_auc()` | **11**, dòng **8** | C**11** L**97**; C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241**; `05_00_Phase5_Run_Order.ipynb` C**5** L**242**; `05_01_LightGBM_Raw.ipynb` C**5** L**311** |
| `save_probabilities()` | **11**, dòng **114** | C**16** L**102**; C**16** L**103**; C**22** L**87**; C**22** L**88** |
| `seed_everything()` | **6**, dòng **81** | C**6** L**109**; C**14** L**165**; C**14** L**166**; `06_Adversarial_XAI.ipynb` C**6** L**281**; `06_Adversarial_XAI.ipynb` C**6** L**282**; `07_Evaluation_Ablation.ipynb` C**6** L**121** |
| `train_one_epoch()` | **14**, dòng **2** | C**14** L**188** |
| `utc_now()` | **11**, dòng **2** | C**11** L**71**; C**11** L**72**; C**14** L**205**; C**14** L**206**; C**19** L**96**; `05_00_Phase5_Run_Order.ipynb` C**5** L**215**; `05_00_Phase5_Run_Order.ipynb` C**5** L**216**; `05_00_Phase5_Run_Order.ipynb` C**6** L**13** (+5) |

## Phase 5 — Model Zoo & Ensemble


### `05_00_Phase5_Run_Order.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `ensure_package()` | **5**, dòng **118** | `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **198** | C**5** L**276**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_01_LightGBM_Raw.ipynb` C**5** L**346**; `05_02_XGBoost_Raw.ipynb` C**5** L**346** |
| `load_raw_arrays()` | **5**, dòng **144** | `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `read_json()` | **5**, dòng **132** | C**6** L**2**; `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **186** | C**5** L**243**; C**5** L**244**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_01_LightGBM_Raw.ipynb` C**5** L**313** |
| `safe_roc_auc()` | **5**, dòng **174** | C**5** L**241**; C**5** L**242**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_01_LightGBM_Raw.ipynb` C**5** L**311** |
| `save_probability()` | **5**, dòng **260** | C**5** L**277**; C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347**; `05_01_LightGBM_Raw.ipynb` C**5** L**348**; `05_02_XGBoost_Raw.ipynb` C**5** L**347** |
| `utc_now()` | **5**, dòng **112** | C**5** L**215**; C**5** L**216**; C**6** L**13**; C**6** L**14**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **270** | `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_01_LightGBM_Raw.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `configure_seed_artifacts()` | **5**, dòng **109** | C**5** L**175**; C**5** L**176**; C**7** L**3**; C**7** L**4**; `05_02_XGBoost_Raw.ipynb` C**5** L**175**; `05_02_XGBoost_Raw.ipynb` C**5** L**176**; `05_02_XGBoost_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_02_XGBoost_Raw.ipynb` C**5** L**346** |
| `load_raw_arrays()` | **5**, dòng **214** | C**6** L**6**; C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6**; `05_02_XGBoost_Raw.ipynb` C**6** L**7**; `05_03_MLP_Raw.ipynb` C**6** L**10** |
| `read_json()` | **5**, dòng **202** | `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_02_XGBoost_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **155** | C**5** L**177**; C**5** L**178**; C**7** L**5**; C**7** L**6**; `05_02_XGBoost_Raw.ipynb` C**5** L**177**; `05_02_XGBoost_Raw.ipynb` C**5** L**178**; `05_02_XGBoost_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **163** | `05_03_MLP_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**55**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**177** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**7** L**53**; C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55**; `05_02_XGBoost_Raw.ipynb` C**7** L**56**; `05_03_MLP_Raw.ipynb` C**7** L**131** |

### `05_02_XGBoost_Raw.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `configure_seed_artifacts()` | **5**, dòng **109** | C**5** L**175**; C**5** L**176**; C**7** L**3**; C**7** L**4**; `05_01_LightGBM_Raw.ipynb` C**5** L**175**; `05_01_LightGBM_Raw.ipynb` C**5** L**176**; `05_01_LightGBM_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | C**6** L**2**; `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `load_raw_arrays()` | **5**, dòng **214** | C**6** L**6**; C**6** L**7**; `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_03_MLP_Raw.ipynb` C**6** L**10** |
| `read_json()` | **5**, dòng **202** | `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **155** | C**5** L**177**; C**5** L**178**; C**7** L**5**; C**7** L**6**; `05_01_LightGBM_Raw.ipynb` C**5** L**177**; `05_01_LightGBM_Raw.ipynb` C**5** L**178**; `05_01_LightGBM_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **163** | `05_03_MLP_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**55**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**177** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**7** L**55**; C**7** L**56**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**131** |

### `05_03_MLP_Raw.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__init__()` | **7**, dòng **4** | C**7** L**5**; C**7** L**6**; `04_PSO_Model_Training.ipynb` C**13** L**23**; `04_PSO_Model_Training.ipynb` C**13** L**24**; `04_PSO_Model_Training.ipynb` C**13** L**70** |
| `configure_seed_artifacts()` | **5**, dòng **109** | C**5** L**175**; C**5** L**176**; C**7** L**50**; C**7** L**51**; `05_01_LightGBM_Raw.ipynb` C**5** L**175**; `05_01_LightGBM_Raw.ipynb` C**5** L**176**; `05_01_LightGBM_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; C**7** L**97**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `forward()` | **7**, dòng **22** | *(method/callback — xem mục bổ sung)* |
| `load_raw_arrays()` | **5**, dòng **214** | C**6** L**10**; C**6** L**11**; `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `make_loader()` | **7**, dòng **27** | C**7** L**41**; C**7** L**42**; C**7** L**76**; C**7** L**77**; `04_PSO_Model_Training.ipynb` C**9** L**77**; `04_PSO_Model_Training.ipynb` C**9** L**79**; `04_PSO_Model_Training.ipynb` C**9** L**81** |
| `predict()` | **7**, dòng **36** | C**7** L**95**; C**7** L**130**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**9** L**66**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**9** L**101**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**9** L**66** |
| `read_json()` | **5**, dòng **202** | `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **155** | C**5** L**177**; C**5** L**178**; C**7** L**52**; C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**5** L**177**; `05_01_LightGBM_Raw.ipynb` C**5** L**178**; `05_01_LightGBM_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **163** | C**7** L**54**; C**7** L**55**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**177**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**178**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**9** L**5** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**7** L**131**; C**7** L**132**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__getitem__()` | **8**, dòng **29** | *(method/callback — xem mục bổ sung)* |
| `__init__()` | **8**, dòng **17** | C**8** L**40**; C**8** L**41**; C**8** L**57**; C**8** L**58**; `04_PSO_Model_Training.ipynb` C**13** L**23**; `04_PSO_Model_Training.ipynb` C**13** L**24**; `04_PSO_Model_Training.ipynb` C**13** L**70** |
| `__len__()` | **8**, dòng **27** | *(method/callback — xem mục bổ sung)* |
| `configure_seed_artifacts()` | **5**, dòng **107** | C**5** L**173**; C**5** L**174**; C**9** L**1**; C**9** L**2**; `05_01_LightGBM_Raw.ipynb` C**5** L**175**; `05_01_LightGBM_Raw.ipynb` C**5** L**176**; `05_01_LightGBM_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | C**6** L**10**; `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; C**9** L**68**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `forward()` | **8**, dòng **45** | *(method/callback — xem mục bổ sung)* |
| `infer_columns()` | **7**, dòng **2** | C**7** L**18**; C**7** L**19**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**7** L**18**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**7** L**19**; `05_04c_CNN_BiLSTM_Sequence_seed456.ipynb` C**7** L**18** |
| `load_raw_arrays()` | **5**, dòng **214** | `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `make_loader()` | **8**, dòng **32** | C**9** L**30**; C**9** L**31**; C**9** L**47**; C**9** L**48**; `04_PSO_Model_Training.ipynb` C**9** L**77**; `04_PSO_Model_Training.ipynb` C**9** L**79**; `04_PSO_Model_Training.ipynb` C**9** L**81** |
| `predict()` | **9**, dòng **25** | C**9** L**66**; C**9** L**101**; `05_03_MLP_Raw.ipynb` C**7** L**95**; `05_03_MLP_Raw.ipynb` C**7** L**130**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**9** L**66** |
| `read_json()` | **5**, dòng **202** | C**7** L**15**; `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **153** | C**5** L**175**; C**5** L**176**; C**9** L**3**; C**9** L**4**; `05_01_LightGBM_Raw.ipynb` C**5** L**177**; `05_01_LightGBM_Raw.ipynb` C**5** L**178**; `05_01_LightGBM_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **161** | C**5** L**177**; C**5** L**178**; C**9** L**5**; C**9** L**6**; `05_03_MLP_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**55**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**5** L**177** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**9** L**102**; C**9** L**103**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__getitem__()` | **8**, dòng **29** | *(method/callback — xem mục bổ sung)* |
| `__init__()` | **8**, dòng **17** | C**8** L**40**; C**8** L**41**; C**8** L**57**; C**8** L**58**; `04_PSO_Model_Training.ipynb` C**13** L**23**; `04_PSO_Model_Training.ipynb` C**13** L**24**; `04_PSO_Model_Training.ipynb` C**13** L**70** |
| `__len__()` | **8**, dòng **27** | *(method/callback — xem mục bổ sung)* |
| `configure_seed_artifacts()` | **5**, dòng **107** | C**5** L**173**; C**5** L**174**; C**9** L**1**; C**9** L**2**; `05_01_LightGBM_Raw.ipynb` C**5** L**175**; `05_01_LightGBM_Raw.ipynb` C**5** L**176**; `05_01_LightGBM_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | C**6** L**10**; `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; C**9** L**68**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `forward()` | **8**, dòng **45** | *(method/callback — xem mục bổ sung)* |
| `infer_columns()` | **7**, dòng **2** | C**7** L**18**; C**7** L**19**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**18**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**19**; `05_04c_CNN_BiLSTM_Sequence_seed456.ipynb` C**7** L**18** |
| `load_raw_arrays()` | **5**, dòng **214** | `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `make_loader()` | **8**, dòng **32** | C**9** L**30**; C**9** L**31**; C**9** L**47**; C**9** L**48**; `04_PSO_Model_Training.ipynb` C**9** L**77**; `04_PSO_Model_Training.ipynb` C**9** L**79**; `04_PSO_Model_Training.ipynb` C**9** L**81** |
| `predict()` | **9**, dòng **25** | C**9** L**66**; C**9** L**101**; `05_03_MLP_Raw.ipynb` C**7** L**95**; `05_03_MLP_Raw.ipynb` C**7** L**130**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**9** L**66** |
| `read_json()` | **5**, dòng **202** | C**7** L**15**; `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **153** | C**5** L**175**; C**5** L**176**; C**9** L**3**; C**9** L**4**; `05_01_LightGBM_Raw.ipynb` C**5** L**177**; `05_01_LightGBM_Raw.ipynb` C**5** L**178**; `05_01_LightGBM_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **161** | C**5** L**177**; C**5** L**178**; C**9** L**5**; C**9** L**6**; `05_03_MLP_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**55**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**177** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**9** L**102**; C**9** L**103**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_04c_CNN_BiLSTM_Sequence_seed456.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__getitem__()` | **8**, dòng **29** | *(method/callback — xem mục bổ sung)* |
| `__init__()` | **8**, dòng **17** | C**8** L**40**; C**8** L**41**; C**8** L**57**; C**8** L**58**; `04_PSO_Model_Training.ipynb` C**13** L**23**; `04_PSO_Model_Training.ipynb` C**13** L**24**; `04_PSO_Model_Training.ipynb` C**13** L**70** |
| `__len__()` | **8**, dòng **27** | *(method/callback — xem mục bổ sung)* |
| `configure_seed_artifacts()` | **5**, dòng **107** | C**5** L**173**; C**5** L**174**; C**9** L**1**; C**9** L**2**; `05_01_LightGBM_Raw.ipynb` C**5** L**175**; `05_01_LightGBM_Raw.ipynb` C**5** L**176**; `05_01_LightGBM_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | C**6** L**10**; `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; C**9** L**68**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `forward()` | **8**, dòng **45** | *(method/callback — xem mục bổ sung)* |
| `infer_columns()` | **7**, dòng **2** | C**7** L**18**; C**7** L**19**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**18**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**19**; `05_04b_CNN_BiLSTM_Sequence_seed123.ipynb` C**7** L**18** |
| `load_raw_arrays()` | **5**, dòng **214** | `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `make_loader()` | **8**, dòng **32** | C**9** L**30**; C**9** L**31**; C**9** L**47**; C**9** L**48**; `04_PSO_Model_Training.ipynb` C**9** L**77**; `04_PSO_Model_Training.ipynb` C**9** L**79**; `04_PSO_Model_Training.ipynb` C**9** L**81** |
| `predict()` | **9**, dòng **25** | C**9** L**66**; C**9** L**101**; `05_03_MLP_Raw.ipynb` C**7** L**95**; `05_03_MLP_Raw.ipynb` C**7** L**130**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**9** L**66** |
| `read_json()` | **5**, dòng **202** | C**7** L**15**; `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **153** | C**5** L**175**; C**5** L**176**; C**9** L**3**; C**9** L**4**; `05_01_LightGBM_Raw.ipynb` C**5** L**177**; `05_01_LightGBM_Raw.ipynb` C**5** L**178**; `05_01_LightGBM_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **161** | C**5** L**177**; C**5** L**178**; C**9** L**5**; C**9** L**6**; `05_03_MLP_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**55**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**177** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**9** L**102**; C**9** L**103**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_05_Weighted_Blending.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `build_base_candidates()` | **6**, dòng **2** | C**6** L**20** |
| `configure_seed_artifacts()` | **5**, dòng **109** | C**5** L**175**; C**5** L**176**; C**7** L**7**; C**7** L**8**; `05_01_LightGBM_Raw.ipynb` C**5** L**175**; `05_01_LightGBM_Raw.ipynb` C**5** L**176**; `05_01_LightGBM_Raw.ipynb` C**7** L**3** |
| `ensure_package()` | **5**, dòng **188** | `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **268** | C**5** L**346**; C**6** L**56**; C**7** L**54**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `load_available_probability_maps()` | **6**, dòng **18** | C**7** L**16**; `05_06_Stacking_Calibration.ipynb` C**7** L**11**; `05_Hybrid_Ensemble.ipynb` C**7** L**16** |
| `load_raw_arrays()` | **5**, dòng **214** | C**7** L**1**; C**7** L**2**; `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `read_json()` | **5**, dòng **202** | `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **256** | C**5** L**313**; C**5** L**314**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **244** | C**5** L**311**; C**5** L**312**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **330** | C**5** L**347**; C**5** L**348**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `set_global_seed()` | **5**, dòng **155** | C**5** L**177**; C**5** L**178**; C**7** L**9**; C**7** L**10**; `05_01_LightGBM_Raw.ipynb` C**5** L**177**; `05_01_LightGBM_Raw.ipynb` C**5** L**178**; `05_01_LightGBM_Raw.ipynb` C**7** L**5** |
| `set_torch_seed()` | **5**, dòng **163** | `05_03_MLP_Raw.ipynb` C**7** L**54**; `05_03_MLP_Raw.ipynb` C**7** L**55**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**5** L**177** |
| `simplex_weights()` | **7**, dòng **19** | C**7** L**37**; C**7** L**38** |
| `threshold_sweep()` | **6**, dòng **50** | `05_Hybrid_Ensemble.ipynb` C**8** L**14** |
| `utc_now()` | **5**, dòng **182** | C**5** L**285**; C**5** L**286**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **340** | C**7** L**86**; C**7** L**87**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_06_Stacking_Calibration.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `ensure_package()` | **5**, dòng **118** | `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **198** | C**5** L**276**; C**6** L**52**; C**7** L**61**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `load_available_probability_maps()` | **6**, dòng **16** | C**7** L**11**; `05_05_Weighted_Blending.ipynb` C**7** L**16**; `05_Hybrid_Ensemble.ipynb` C**7** L**16** |
| `load_raw_arrays()` | **5**, dòng **144** | C**7** L**8**; C**7** L**9**; `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `read_json()` | **5**, dòng **132** | `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **186** | C**5** L**243**; C**5** L**244**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **174** | C**5** L**241**; C**5** L**242**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **260** | C**5** L**277**; C**5** L**278**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `stack_matrix()` | **7**, dòng **16** | C**7** L**21** |
| `threshold_sweep()` | **6**, dòng **46** | `05_Hybrid_Ensemble.ipynb` C**8** L**14** |
| `utc_now()` | **5**, dòng **112** | C**5** L**215**; C**5** L**216**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **270** | C**7** L**76**; C**7** L**77**; `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

### `05_Hybrid_Ensemble.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `ensure_package()` | **5**, dòng **118** | `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_predictions()` | **5**, dòng **198** | C**5** L**276**; C**6** L**52**; C**8** L**8**; C**8** L**118**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `load_available_probability_maps()` | **6**, dòng **16** | C**7** L**16**; `05_05_Weighted_Blending.ipynb` C**7** L**16**; `05_06_Stacking_Calibration.ipynb` C**7** L**11** |
| `load_raw_arrays()` | **5**, dòng **144** | C**7** L**1**; C**7** L**2**; `05_01_LightGBM_Raw.ipynb` C**6** L**6**; `05_01_LightGBM_Raw.ipynb` C**6** L**7**; `05_02_XGBoost_Raw.ipynb` C**6** L**6** |
| `read_json()` | **5**, dòng **132** | `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** |
| `safe_pr_auc()` | **5**, dòng **186** | C**5** L**243**; C**5** L**244**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **5**, dòng **174** | C**5** L**241**; C**5** L**242**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `save_probability()` | **5**, dòng **260** | C**5** L**277**; C**5** L**278**; `05_00_Phase5_Run_Order.ipynb` C**5** L**277**; `05_00_Phase5_Run_Order.ipynb` C**5** L**278**; `05_01_LightGBM_Raw.ipynb` C**5** L**347** |
| `threshold_sweep()` | **6**, dòng **46** | C**8** L**14** |
| `utc_now()` | **5**, dòng **112** | C**5** L**215**; C**5** L**216**; C**8** L**81**; C**8** L**82**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** |
| `write_metrics()` | **5**, dòng **270** | `05_01_LightGBM_Raw.ipynb` C**7** L**53**; `05_01_LightGBM_Raw.ipynb` C**7** L**54**; `05_02_XGBoost_Raw.ipynb` C**7** L**55** |

## Phase 6 — Adversarial & XAI


### `06_Adversarial_XAI.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__init__()` | **16**, dòng **22** | C**16** L**23**; C**16** L**24**; C**16** L**44**; C**16** L**45**; `04_PSO_Model_Training.ipynb` C**13** L**23**; `04_PSO_Model_Training.ipynb` C**13** L**24**; `04_PSO_Model_Training.ipynb` C**13** L**70** |
| `attack_in_batches()` | **20**, dòng **64** | C**20** L**110**; C**20** L**116** |
| `build_model()` | **16**, dòng **88** | C**16** L**135**; `04_PSO_Model_Training.ipynb` C**14** L**168** |
| `build_raw_feature_metadata()` | **24**, dòng **2** | C**24** L**72** |
| `build_raw_feature_names()` | **8**, dòng **2** | C**8** L**100** |
| `build_stack_features()` | **17**, dòng **64** | C**17** L**128** |
| `choose_case()` | **6**, dòng **262** | C**12** L**30**; C**26** L**48** |
| `clamp_features()` | **20**, dòng **2** | C**20** L**24**; C**20** L**58** |
| `ensure_package()` | **5**, dòng **24** | C**5** L**38**; C**5** L**40**; C**5** L**42**; C**5** L**44**; C**5** L**46**; `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `evaluate_condition()` | **20**, dòng **88** | C**20** L**105**; C**20** L**106**; C**20** L**111**; C**20** L**112**; C**20** L**117** (+1) |
| `evaluate_predictions()` | **19**, dòng **26** | C**20** L**95**; C**20** L**96**; C**20** L**97**; C**20** L**98**; `04_PSO_Model_Training.ipynb` C**14** L**122**; `05_00_Phase5_Run_Order.ipynb` C**5** L**276**; `05_01_LightGBM_Raw.ipynb` C**5** L**346** |
| `expand_component_block_weights()` | **16**, dòng **2** | C**16** L**51**; `04_PSO_Model_Training.ipynb` C**13** L**77** |
| `extract_components_matrix()` | **24**, dòng **46** | C**24** L**70** |
| `fgsm_attack_batch()` | **20**, dòng **8** | C**20** L**73**; C**20** L**74** |
| `find_selected_stacker_model()` | **17**, dòng **70** | C**17** L**124** |
| `forward()` | **16**, dòng **29** | *(method/callback — xem mục bổ sung)* |
| `pgd_attack_batch()` | **20**, dòng **30** | C**20** L**77**; C**20** L**78** |
| `positive_class_probability()` | **6**, dòng **238** | C**8** L**129**; C**17** L**55**; C**17** L**56**; C**17** L**57**; C**17** L**58** (+1) |
| `predict_base_probas()` | **17**, dòng **48** | C**17** L**84** |
| `predict_dl_proba()` | **17**, dòng **28** | C**17** L**53**; C**17** L**54**; C**17** L**148**; C**20** L**90** |
| `predict_final_ensemble_fake_proba()` | **17**, dòng **82** | C**17** L**142**; C**17** L**150**; C**20** L**92**; C**26** L**90** |
| `predict_final_ensemble_proba()` | **17**, dòng **140** | *(method/callback — xem mục bổ sung)* |
| `predict_xgb_raw_fake_proba()` | **8**, dòng **127** | C**8** L**134**; C**12** L**139** |
| `predict_xgb_raw_proba()` | **8**, dòng **132** | C**12** L**124**; C**12** L**125** |
| `read_json()` | **6**, dòng **194** | C**8** L**35**; C**8** L**37**; C**14** L**47**; C**14** L**49**; C**14** L**51**; `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** (+2) |
| `record_fallback()` | **6**, dòng **202** | C**8** L**103**; C**8** L**104**; C**10** L**136**; C**10** L**137**; C**10** L**173** (+25) |
| `safe_pr_auc()` | **19**, dòng **14** | C**19** L**77**; C**19** L**78**; `04_PSO_Model_Training.ipynb` C**11** L**99**; `04_PSO_Model_Training.ipynb` C**11** L**100**; `05_00_Phase5_Run_Order.ipynb` C**5** L**243** |
| `safe_roc_auc()` | **19**, dòng **2** | C**19** L**75**; C**19** L**76**; `04_PSO_Model_Training.ipynb` C**11** L**97**; `04_PSO_Model_Training.ipynb` C**11** L**98**; `05_00_Phase5_Run_Order.ipynb` C**5** L**241** |
| `seed_everything()` | **6**, dòng **166** | C**6** L**281**; C**6** L**282**; `04_PSO_Model_Training.ipynb` C**6** L**109**; `04_PSO_Model_Training.ipynb` C**14** L**165**; `04_PSO_Model_Training.ipynb` C**14** L**166** |
| `select_final_lime_cases()` | **12**, dòng **2** | C**12** L**74** |
| `select_lime_cases()` | **26**, dòng **20** | C**26** L**92** |
| `stratified_subset_indices()` | **6**, dòng **222** | C**10** L**2**; C**12** L**70**; C**19** L**92**; C**23** L**4**; C**26** L**86** |
| `to_two_column_prob()` | **6**, dòng **254** | C**8** L**134**; C**17** L**142** |
| `torch_load_checkpoint()` | **16**, dòng **110** | C**16** L**122** |
| `utc_now()` | **6**, dòng **188** | C**6** L**207**; C**6** L**208**; C**8** L**74**; C**8** L**75**; C**10** L**56**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** (+31) |

## Phase 7 — Evaluation & Ablation


### `07_Evaluation_Ablation.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `array_memory_mb()` | **15**, dòng **64** | C**15** L**105**; C**19** L**96**; `02_Feature_Engineering.ipynb` C**20** L**44**; `02_Feature_Engineering.ipynb` C**20** L**46**; `03_PCA_Feature_Selection.ipynb` C**8** L**48** |
| `build_basic_behavioral_features()` | **18**, dòng **152** | C**19** L**141** |
| `controlled_train_eval()` | **18**, dòng **72** | C**19** L**66**; C**19** L**67**; C**19** L**105**; C**19** L**106**; C**19** L**156** (+1) |
| `csv_rows()` | **9**, dòng **16** | C**9** L**166** |
| `ensure_package()` | **5**, dòng **28** | C**5** L**42**; C**5** L**44**; `05_01_LightGBM_Raw.ipynb` C**6** L**2**; `05_02_XGBoost_Raw.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**6** L**10** |
| `environment_versions()` | **6**, dòng **102** | C**25** L**137**; C**25** L**138**; `04_PSO_Model_Training.ipynb` C**23** L**89**; `04_PSO_Model_Training.ipynb` C**23** L**90** |
| `error_case_table()` | **22**, dòng **50** | C**22** L**116**; C**22** L**118** |
| `evaluate_probabilities()` | **11**, dòng **22** | C**11** L**185**; C**11** L**186**; C**11** L**211**; C**11** L**212**; C**11** L**237** (+7) |
| `evaluate_test_probability()` | **18**, dòng **2** | C**19** L**9**; C**19** L**28**; C**19** L**44**; C**19** L**226**; C**19** L**243** |
| `fit_lightgbm_predict()` | **15**, dòng **54** | C**15** L**118** |
| `lightgbm_config()` | **15**, dòng **18** | C**15** L**56**; C**18** L**106** |
| `load_labels()` | **11**, dòng **16** | C**11** L**154**; C**26** L**86**; C**26** L**88** |
| `load_multiseed_blend_probs()` | **26**, dòng **18** | C**26** L**93**; C**26** L**94** |
| `load_prob()` | **11**, dòng **2** | C**11** L**174**; C**11** L**175** |
| `load_test_context()` | **22**, dòng **2** | C**22** L**114** |
| `multiseed_prediction_dir()` | **26**, dòng **8** | C**26** L**20** |
| `npy_shape_dtype()` | **9**, dòng **2** | C**9** L**161**; C**9** L**162** |
| `read_json()` | **6**, dòng **94** | C**9** L**220**; C**9** L**222**; C**9** L**224**; C**9** L**226**; C**9** L**228**; `04_PSO_Model_Training.ipynb` C**8** L**43**; `05_00_Phase5_Run_Order.ipynb` C**6** L**2**; `05_04a_CNN_BiLSTM_Sequence_seed42.ipynb` C**7** L**15** (+2) |
| `seed_everything()` | **6**, dòng **78** | C**6** L**121**; C**6** L**122**; `04_PSO_Model_Training.ipynb` C**6** L**109**; `04_PSO_Model_Training.ipynb` C**14** L**165**; `04_PSO_Model_Training.ipynb` C**14** L**166** |
| `select_multiseed_thresholds()` | **26**, dòng **34** | C**26** L**100**; C**26** L**101** |
| `selected_weighted_blend()` | **18**, dòng **26** | *(method/callback — xem mục bổ sung)* |
| `stratified_sample_indices()` | **15**, dòng **2** | C**15** L**91**; C**18** L**94** |
| `train_only_reduce_features()` | **18**, dòng **208** | C**19** L**143** |
| `utc_now()` | **6**, dòng **88** | C**9** L**171**; C**9** L**172**; C**11** L**99**; C**11** L**100**; C**12** L**21**; `04_PSO_Model_Training.ipynb` C**11** L**71**; `04_PSO_Model_Training.ipynb` C**11** L**72**; `04_PSO_Model_Training.ipynb` C**14** L**205** (+9) |

## Phase 9 — Diagnostic


### `tests/01_DL_PCA_Diagnostic_Test.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `__init__()` | **6**, dòng **4** | C**6** L**5**; C**6** L**6**; C**6** L**38**; C**6** L**39**; `04_PSO_Model_Training.ipynb` C**13** L**23**; `04_PSO_Model_Training.ipynb` C**13** L**24**; `04_PSO_Model_Training.ipynb` C**13** L**70** |
| `choose_threshold_by_precision()` | **4**, dòng **168** | C**6** L**262**; C**6** L**263**; C**9** L**54**; C**9** L**55** |
| `fmt()` | **4**, dòng **116** | C**10** L**24**; C**10** L**25**; C**10** L**36**; C**10** L**37**; C**10** L**49**; `08_Final_Report_Kaggle.ipynb` C**12** L**4**; `08_Final_Report_Kaggle.ipynb` C**12** L**6**; `08_Final_Report_Kaggle.ipynb` C**12** L**8** (+12) |
| `forward()` | **6**, dòng **29** | *(method/callback — xem mục bổ sung)* |
| `load_array()` | **4**, dòng **74** | C**5** L**2**; C**5** L**4**; C**5** L**6**; C**5** L**8**; C**5** L**10** (+11) |
| `load_csv()` | **4**, dòng **84** | C**5** L**21**; C**5** L**23** |
| `load_json()` | **4**, dòng **94** | C**5** L**25** |
| `make_loader()` | **6**, dòng **104** | C**6** L**118**; C**6** L**142**; C**6** L**144**; `04_PSO_Model_Training.ipynb` C**9** L**77**; `04_PSO_Model_Training.ipynb` C**9** L**79**; `04_PSO_Model_Training.ipynb` C**9** L**81** |
| `metric_bundle()` | **4**, dòng **132** | C**4** L**176**; C**6** L**208**; C**6** L**265**; C**6** L**267**; C**7** L**84** (+5) |
| `predict_probs()` | **6**, dòng **114** | C**6** L**251**; C**6** L**253** |
| `save_csv()` | **4**, dòng **106** | C**5** L**59**; C**5** L**60**; C**5** L**110**; C**5** L**111**; C**6** L**260** (+15) |
| `train_model()` | **6**, dòng **136** | C**7** L**20**; C**7** L**45**; C**8** L**2** |

## Benchmark & Viz


### `10_Baseline_Algorithm_Benchmark.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `build_estimator()` | **4**, dòng **23** | C**5** L**11** |
| `compute_metrics()` | **4**, dòng **40** | C**5** L**21** |
| `tim_thu_muc_goc()` | **3**, dòng **2** | C**3** L**33**; `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb` C**4** L**53** |

### `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb`

| Hàm | Định nghĩa (Cell, Dòng) | Gọi tại (Cell, Dòng) |
|-----|------------------------|----------------------|
| `dam_bao_thu_vien()` | **3**, dòng **11** | C**3** L**30**; C**3** L**31** |
| `ghep_metric_tu_nhieu_file()` | **6**, dòng **28** | C**6** L**176** |
| `in_insight()` | **6**, dòng **165** | C**10** L**13**; C**10** L**14**; C**14** L**34**; C**14** L**35**; C**18** L**22** (+13) |
| `loc_metric_test()` | **6**, dòng **4** | C**6** L**58** |
| `tim_thu_muc_goc()` | **4**, dòng **7** | C**4** L**53**; `10_Baseline_Algorithm_Benchmark.ipynb` C**3** L**33** |
| `ve_luu_va_hien_thi()` | **6**, dòng **142** | C**8** L**40**; C**8** L**41**; C**12** L**35**; C**12** L**36**; C**16** L**64** (+11) |

---

## Gọi thư viện thuật toán (không bọc hàm riêng)

| Notebook | Thuật toán | Cell | Dòng | Code |
|----------|------------|------|------|------|
| `04_PSO_Model_Training.ipynb` | PSO import | **5** | **13** | `# import subprocess/sys: cài pyswarm nếu thiếu` |
| `04_PSO_Model_Training.ipynb` | PSO import | **5** | **59** | `# PYSWARM_AVAILABLE: cờ thư viện pyswarm đã cài thành công` |
| `04_PSO_Model_Training.ipynb` | PSO import | **5** | **60** | `PYSWARM_AVAILABLE = True` |
| `02_Feature_Engineering.ipynb` | ModernBERT forward | **10** | **5** | `# from transformers import AutoModel, AutoTokenizer: Hugging` |
| `02_Feature_Engineering.ipynb` | ModernBERT forward | **10** | **6** | `from transformers import AutoModel, AutoTokenizer` |
| `02_Feature_Engineering.ipynb` | ModernBERT forward | **10** | **17** | `# mean_pool_last_hidden_state: mean pooling có mask attentio` |
| `03_PCA_Feature_Selection.ipynb` | PCA/SVD fit | **5** | **24** | `# PCA, IncrementalPCA, TruncatedSVD: các backend giảm chiều` |
| `03_PCA_Feature_Selection.ipynb` | PCA/SVD fit | **5** | **25** | `from sklearn.decomposition import PCA, IncrementalPCA, Trunc` |
| `03_PCA_Feature_Selection.ipynb` | PCA/SVD fit | **5** | **61** | `# SVD_MAX_COMPONENTS: giới hạn component khi dùng TruncatedS` |
| `05_01_LightGBM_Raw.ipynb` | LGBM fit/predict | **6** | **3** | `# from lightgbm import LGBMClassifier: thư viện LightGBM` |
| `05_01_LightGBM_Raw.ipynb` | LGBM fit/predict | **6** | **4** | `from lightgbm import LGBMClassifier` |
| `05_01_LightGBM_Raw.ipynb` | LGBM fit/predict | **7** | **42** | `model = LGBMClassifier(**CONFIG)` |
| `05_02_XGBoost_Raw.ipynb` | XGB fit/predict | **6** | **3** | `# from xgboost import XGBClassifier: thư viện XGBoost` |
| `05_02_XGBoost_Raw.ipynb` | XGB fit/predict | **6** | **4** | `from xgboost import XGBClassifier` |
| `05_02_XGBoost_Raw.ipynb` | XGB fit/predict | **7** | **44** | `model = XGBClassifier(**CONFIG)` |
| `05_06_Stacking_Calibration.ipynb` | Stacking fit | **7** | **1** | `# from sklearn.calibration import CalibratedClassifierCV: th` |
| `05_06_Stacking_Calibration.ipynb` | Stacking fit | **7** | **2** | `from sklearn.calibration import CalibratedClassifierCV` |
| `05_06_Stacking_Calibration.ipynb` | Stacking fit | **7** | **36** | `# model.fit(X_stack["train"], y["train"]): fit model/reducer` |
| `06_Adversarial_XAI.ipynb` | SHAP | **10** | **11** | `final_explainer = shap.TreeExplainer(xgb_raw_model['model'])` |
| `06_Adversarial_XAI.ipynb` | SHAP | **10** | **89** | `# final_shap_importance_df.groupby("feature_group", as_index` |
| `06_Adversarial_XAI.ipynb` | SHAP | **10** | **147** | `# shap.summary_plot(: tính tổng` |
| `06_Adversarial_XAI.ipynb` | LIME | **12** | **121** | `explanation = final_lime_explainer.explain_instance(` |
| `06_Adversarial_XAI.ipynb` | LIME | **26** | **129** | `explanation = explainer.explain_instance(` |
| `10_Baseline_Algorithm_Benchmark.ipynb` | Sklearn fit | **5** | **12** | `# est.fit(x_train, y_train): fit model/reducer trên dữ liệu ` |
| `10_Baseline_Algorithm_Benchmark.ipynb` | Sklearn fit | **5** | **13** | `est.fit(x_train, y_train)` |
| `10_Baseline_Algorithm_Benchmark.ipynb` | Sklearn fit | **5** | **15** | `y_prob = est.predict_proba(x_test)[:, 1]` |

---

## Bổ sung — lời gọi quan trọng (đã xác minh thủ công)

| Notebook | Hàm | Loại | Cell | Dòng |
|----------|-----|------|------|------|
| `02_Feature_Engineering.ipynb` | `load_bert_model()` | Định nghĩa | **10** | **32** |
| `02_Feature_Engineering.ipynb` | `load_bert_model()` | Gọi | **11** | **2** |
| `02_Feature_Engineering.ipynb` | `extract_or_load_bert_embeddings()` | Gọi train | **11** | **16** |
| `02_Feature_Engineering.ipynb` | `extract_or_load_bert_embeddings()` | Gọi val | **11** | **22** |
| `02_Feature_Engineering.ipynb` | `extract_or_load_bert_embeddings()` | Gọi test | **11** | **28** |
| `02_Feature_Engineering.ipynb` | `add_basic_behavioral_features()` | Gọi (dict comp) | **15** | **8** |
| `02_Feature_Engineering.ipynb` | `fit_reviewer_behavior_map()` | Gọi | **16** | **2** |
| `02_Feature_Engineering.ipynb` | `add_advanced_behavioral_features()` | Gọi (dict comp) | **17** | **8** |
| `04_PSO_Model_Training.ipynb` | `pso_objective()` | Callback PSO | **20** | **8** |
| `04_PSO_Model_Training.ipynb` | `fallback_pso()` | Gọi pmin | **20** | **120** |
| `04_PSO_Model_Training.ipynb` | `normalize_pso_result()` | Gọi | **20** | **155** |
| `04_PSO_Model_Training.ipynb` | `predict_probabilities()` | Gọi trong evaluate_model | **14** | **118** |
| `05_03_MLP_Raw.ipynb` | `predict()` | Gọi val mỗi epoch | **7** | **97** |
| `05_03_MLP_Raw.ipynb` | `predict()` | Gọi prob_map | **7** | **132** |
| `05_04a_*` | `predict()` | Gọi prob_map | **9** | **103** |
| `05_05_Weighted_Blending.ipynb` | `threshold_sweep()` | Gọi sweep | **6** | **56** |
| `05_05_Weighted_Blending.ipynb` | `threshold_sweep()` | Gọi grid | **7** | **54** |
| `05_06_Stacking_Calibration.ipynb` | `stack_matrix()` | Gọi build X_stack | **7** | **4** |
| `05_Hybrid_Ensemble.ipynb` | `threshold_sweep()` | Gọi | **8** | **14** |
| `06_Adversarial_XAI.ipynb` | `fgsm_attack_batch()` | Gọi | **20** | **13** |
| `06_Adversarial_XAI.ipynb` | `pgd_attack_batch()` | Gọi | **20** | **17** |
| `06_Adversarial_XAI.ipynb` | `evaluate_condition()` | Gọi clean/fgsm/pgd | **20** | **45** |
| `06_Adversarial_XAI.ipynb` | `select_final_lime_cases()` | Gọi | **13** | **2** |
| `06_Adversarial_XAI.ipynb` | `select_lime_cases()` | Gọi | **27** | **2** |
| `07_Evaluation_Ablation.ipynb` | `selected_weighted_blend()` | Gọi ablation | **19** | **60** |
| `07_Evaluation_Ablation.ipynb` | `fit_lightgbm_predict()` | Gọi CV fold | **15** | **110** |
| `07_Evaluation_Ablation.ipynb` | `error_case_table()` | Gọi | **23** | **2** |
| `07_Evaluation_Ablation.ipynb` | `load_multiseed_blend_probs()` | Gọi | **27** | **8** |
| `07_Evaluation_Ablation.ipynb` | `select_multiseed_thresholds()` | Gọi | **27** | **15** |
| `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb` | `ghep_metric_tu_nhieu_file()` | Gọi | **7** | **2** |
| `09_Bieu_Do_So_Sanh_Mo_Hinh.ipynb` | `loc_metric_test()` | Gọi trong ghep_metric | **6** | **35** |