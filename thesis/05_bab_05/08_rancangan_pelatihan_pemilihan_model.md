## **5.8 Implementasi Pelatihan dan Pemilihan Model**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.8, yaitu pemuatan data latih, pemeriksaan komponen utama, pencarian hyperparamater dengan validasi silang, dan pembentukan model akhir dari pembagian terbaik. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.8, sedangkan grid dan pemanggilan untuk setiap algoritma disajikan pada Subbab 5.9 hingga 5.13.

Kode Program 5.55 menetapkan pengaturan pencarian hyperparamater. Konstanta *N_SPLITS* dan *VALIDATION_SIZE* menetapkan lima pembagian dengan bagian validasi 20%. Kamus *GRID_SEARCH_SCORING* menetapkan empat metrik penilaian, yaitu F1-*score*, presisi, dan *recall* rata-rata makro dengan *zero_division=0*, serta akurasi, sedangkan *GRID_SEARCH_RANK_METRIC* menetapkan F1-*score* rata-rata makro sebagai kriteria pemilihan. Konstanta *GRID_SEARCH_N_JOBS* bernilai satu sehingga pelatihan pada pencarian dijalankan secara berurutan. Konstanta *USE_SMOTE* menentukan pengaturan yang sedang dijalankan, yaitu *False* untuk pengaturan tanpa SMOTE dan *True* untuk pengaturan dengan SMOTE, sedangkan *SMOTE_K_NEIGHBORS_GRID* menetapkan nilai jumlah tetangga SMOTE yang dicari.

```python
N_SPLITS = 5
VALIDATION_SIZE = 0.2

GRID_SEARCH_SCORING = {
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "accuracy": "accuracy",
}
GRID_SEARCH_RANK_METRIC = "f1_macro"
GRID_SEARCH_N_JOBS = 1

USE_SMOTE = False
SMOTE_K_NEIGHBORS_GRID = [3, 5, 7]
```

Kode Program 5.55 Pengaturan pencarian hyperparamater

Kode Program 5.56 mengimplementasikan Langkah 1. Fungsi *load_labels* membaca kolom label dari file *encoded-labels.parquet* sebuah himpunan. Fungsi *load_split* membaca fitur secara *lazy*, memuatnya per kelompok 500.000 baris ke satu matriks *float32* menggunakan *load_features* (pendahuluan Bab 5), memuat labelnya, dan menghentikan proses dengan galat apabila jumlah baris keduanya berbeda. Sel terakhir memuat data latih dan menampilkan jumlah baris dan fiturnya.

```python
def load_labels(
    split_folder_path: str,
    labels_file_name: str = ENCODED_LABELS_FILE_NAME,
    label_column: str = LABEL_COLUMN,
) -> pd.Series:
    labels = pl.scan_parquet(os.path.join(split_folder_path, labels_file_name)).select(label_column)
    return labels.collect().to_series().to_pandas()

def load_split(
    split_folder_path: str,
    features_file_name: str = FEATURES_FILE_NAME,
    labels_file_name: str = ENCODED_LABELS_FILE_NAME,
    label_column: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    features = pl.scan_parquet(os.path.join(split_folder_path, features_file_name))
    features = load_features(features, f"Loading {split_folder_path}")
    labels = load_labels(split_folder_path, labels_file_name, label_column)
    if len(features) != len(labels):
        raise ValueError(f"{split_folder_path}: {len(features):,} feature rows but {len(labels):,} labels")
    return features, labels

train_features, train_labels = load_split(PATH_FOLDER_SPLITTED_DATASET_TRAIN)
end_status(f"Loaded {len(train_labels):,} train rows x {train_features.shape[1]} features")
```

Kode Program 5.56 Pemuatan data latih

Kode Program 5.57 mengimplementasikan Langkah 2. Fungsi *fit_principal_components* melatih tahap praproses *pipeline*, yaitu konversi tipe data, standardisasi, dan PCA, pada seluruh data latih dan mengembalikan objek PCA yang telah dilatih. Fungsi *summarize_explained_variance* menyusun rasio varians yang dijelaskan oleh setiap komponen beserta varians kumulatifnya dalam persen. Sel terakhir menampilkan jumlah komponen yang terpilih dan varians kumulatif yang dicapainya.

```python
COMPONENT_COLUMN = "component"
EXPLAINED_VARIANCE_COLUMN = "explained_variance_percentage"
CUMULATIVE_VARIANCE_COLUMN = "cumulative_variance_percentage"

def fit_principal_components(features: pd.DataFrame) -> PCA:
    print_status(f"Fitting the scaler and PCA on {len(features):,} rows")
    preprocessing = Pipeline(make_preprocessing_steps()).fit(features)
    return preprocessing.named_steps["pca"]

def summarize_explained_variance(pca: PCA) -> pd.DataFrame:
    component_count = len(pca.explained_variance_ratio_)
    return pd.DataFrame({
        COMPONENT_COLUMN: np.arange(1, component_count + 1),
        EXPLAINED_VARIANCE_COLUMN: pca.explained_variance_ratio_ * 100,
        CUMULATIVE_VARIANCE_COLUMN: np.cumsum(pca.explained_variance_ratio_) * 100,
    })

pca = fit_principal_components(train_features)
results = summarize_explained_variance(pca)
kept_variance = results[CUMULATIVE_VARIANCE_COLUMN].iloc[-1]
end_status(
    f"{pca.n_components_} principal components keep {kept_variance:.2f}% of the variance "
    f"(more than {PCA_VARIANCE_THRESHOLD:.0%})"
)
with pd.option_context("display.max_rows", None):
    display(results)
del pca, results, kept_variance
```

Kode Program 5.57 Pemeriksaan jumlah komponen utama pada data latih

Fungsi *make_param_grid* pada Kode Program 5.58 mengimplementasikan Langkah 3. Fungsi ini menerima grid hyperparamater sebuah algoritma, baik berupa satu kamus maupun daftar kamus, kemudian menambahkan jumlah tetangga SMOTE (*smote__k_neighbors*) ke setiap kamus apabila pengaturan dengan SMOTE digunakan. Daftar kamus digunakan ketika grid terdiri atas beberapa bagian dengan hyperparamater yang berbeda, seperti pada SVM (Subbab 5.10).

```python
def make_param_grid(
    model_param_grid: dict | list[dict],
    use_smote: bool = True,
    smote_k_neighbors: list[int] = SMOTE_K_NEIGHBORS_GRID,
) -> list[dict]:
    if isinstance(model_param_grid, dict):
        model_param_grid = [model_param_grid]

    shared_param_grid = {}
    if use_smote:
        shared_param_grid["smote__k_neighbors"] = smote_k_neighbors

    param_grid = []
    for model_params in model_param_grid:
        param_grid.append({**shared_param_grid, **model_params})
    return param_grid
```

Kode Program 5.58 Penyusunan grid hyperparamater

Kode Program 5.59 memuat dua fungsi bantu validasi silang. Fungsi *make_splitter* membentuk *StratifiedShuffleSplit* dengan lima pembagian, bagian validasi 20%, dan *random state* 42. Karena *random state*-nya tetap, setiap pemanggilan menghasilkan lima pembagian yang sama, sehingga pembagian dapat dibentuk ulang pada saat model akhir dilatih. Fungsi *get_split_score_columns* menyusun nama kolom skor setiap pembagian pada hasil *GridSearchCV*, misalnya *split0_test_f1_macro*.

```python
def make_splitter(
    n_splits: int = N_SPLITS,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> StratifiedShuffleSplit:
    return StratifiedShuffleSplit(n_splits=n_splits, test_size=validation_size, random_state=random_state)

def get_split_score_columns(rank_metric: str = GRID_SEARCH_RANK_METRIC, n_splits: int = N_SPLITS) -> list[str]:
    columns = []
    for split_index in range(n_splits):
        columns.append(f"split{split_index}_test_{rank_metric}")
    return columns
```

Kode Program 5.59 Pembentukan pembagian validasi silang

Fungsi *run_grid_search* pada Kode Program 5.60 mengimplementasikan Langkah 4. Fungsi ini membentuk *GridSearchCV* dengan *pipeline*, grid, empat metrik penilaian, dan pembagian dari *make_splitter*. Parameter *refit=False* membuat *GridSearchCV* tidak melatih ulang model terbaik secara otomatis, karena model akhir dibentuk secara terpisah. Jumlah seluruh pelatihan dihitung dari jumlah kombinasi dikalikan jumlah pembagian, dan objek *FitProgress* diteruskan ke setiap pelatihan sehingga kemajuan pencarian dapat dipantau. Kombinasi yang gagal dilatih memperoleh skor kosong karena *GridSearchCV* menggunakan nilai bawaan *error_score=nan*. Hasil pencarian disusun menjadi tabel yang diurutkan menurut peringkat F1-*score* makro, dan jumlah pelatihan yang gagal dihitung dari skor kosong pada kolom skor setiap pembagian.

```python
def run_grid_search(
    model_name: str,
    pipeline: Pipeline,
    param_grid: list[dict],
    features: pd.DataFrame,
    labels: pd.Series,
    scoring: dict = GRID_SEARCH_SCORING,
    rank_metric: str = GRID_SEARCH_RANK_METRIC,
    n_jobs: int = GRID_SEARCH_N_JOBS,
) -> pd.DataFrame:
    splitter = make_splitter()
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit=False,  # the best fold model is refitted separately, so a failed refit cannot lose the results
        cv=splitter,
        n_jobs=n_jobs,
    )
    fit_count = len(ParameterGrid(param_grid)) * splitter.get_n_splits()
    progress = FitProgress(fit_count, f"{model_name}: grid search on {len(labels):,} rows")
    grid_search.fit(features, labels, progress=progress)

    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(f"rank_test_{rank_metric}", ignore_index=True)

    elapsed = format_duration(time.time() - progress.started)
    failed_count = results[get_split_score_columns(rank_metric)].isna().sum().sum()
    best_score = results.loc[0, f"mean_test_{rank_metric}"]
    end_status(f"{model_name}: {fit_count} fits in {elapsed}, {failed_count} failed, best {rank_metric} {best_score:.4f}")
    return results
```

Kode Program 5.60 Pencarian hyperparamater dengan GridSearchCV

Kode Program 5.61 memuat fungsi penyimpanan dan peringkasan hasil pencarian. Fungsi *save_grid_search_results* menyimpan seluruh hasil pencarian sebagai file CSV dan kombinasi terbaik sebagai file JSON pada folder *grid-search* sesuai pengaturan SMOTE. Fungsi *summarize_grid_search* memilih kolom hyperparamater, rata-rata dan simpangan baku skor setiap metrik, serta rata-rata waktu pelatihan untuk ditampilkan. Fungsi *search_model* menyatukan penyusunan *pipeline*, penyusunan grid, pencarian, dan penyimpanan hasil untuk satu algoritma.

```python
def save_grid_search_results(results: pd.DataFrame, model_name: str, use_smote: bool) -> None:
    if use_smote:
        folder_path = PATH_FOLDER_GRID_SEARCH_WITH_SMOTE
    else:
        folder_path = PATH_FOLDER_GRID_SEARCH_WOUT_SMOTE
    os.makedirs(folder_path, exist_ok=True)

    results.to_csv(os.path.join(folder_path, f"{model_name}.csv"), index=False)
    with open(os.path.join(folder_path, f"{model_name}-best-params.json"), "w", encoding="utf-8") as file:
        json.dump(results.loc[0, "params"], file, indent=4, default=repr)

def summarize_grid_search(results: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for column in results.columns:
        if column.startswith(("param_", "mean_test_", "std_test_")):
            columns.append(column)
    columns.append("mean_fit_time")
    return results[columns]

def search_model(
    model_name: str,
    make_pipeline: Callable[[bool], Pipeline],
    model_param_grid: dict | list[dict],
    features: pd.DataFrame,
    labels: pd.Series,
    use_smote: bool,
) -> pd.DataFrame:
    results = run_grid_search(
        model_name,
        make_pipeline(use_smote),
        make_param_grid(model_param_grid, use_smote),
        features,
        labels,
    )
    save_grid_search_results(results, model_name, use_smote)
    return results
```

Kode Program 5.61 Penyimpanan dan peringkasan hasil pencarian

Fungsi *save_best_fold_model* pada Kode Program 5.62 mengimplementasikan Langkah 5 hingga 7. Fungsi ini mengambil skor F1-*score* makro kombinasi terbaik pada kelima pembagian dan memilih pembagian dengan skor tertinggi. Pembagian tersebut dibentuk ulang dengan *make_splitter*, kemudian *pipeline* baru disusun dengan hyperparamater terbaik, dilatih pada bagian latih pembagian tersebut, dan dinilai pada bagian validasinya. Hyperparamater disalin dengan *clone* sebelum diterapkan, karena hyperparamater dapat berupa objek model, misalnya *LinearSVC* pada SVM, yang tidak boleh berbagi keadaan dengan objek pada hasil pencarian. *Pipeline* yang telah dilatih disimpan pada folder *trained-models* sesuai pengaturan SMOTE, dan jumlah komponen utama serta skor validasinya ditampilkan bersama skor pembagian yang sama pada pencarian sebagai pemeriksaan konsistensi.

```python
def save_best_fold_model(
    model_name: str,
    make_pipeline: Callable[[bool], Pipeline],
    results: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    use_smote: bool,
    scoring: dict = GRID_SEARCH_SCORING,
    rank_metric: str = GRID_SEARCH_RANK_METRIC,
) -> None:
    split_scores = results.loc[0, get_split_score_columns(rank_metric)].to_numpy(dtype=float)
    best_split = int(np.nanargmax(split_scores))
    train_rows, validation_rows = list(make_splitter().split(features, labels))[best_split]

    print_status(f"{model_name}: fitting the best params on split {best_split} ({len(train_rows):,} rows)")
    pipeline = make_pipeline(use_smote).set_params(**clone(results.loc[0, "params"], safe=False))
    pipeline.fit(features.iloc[train_rows], labels.iloc[train_rows])

    print_status(f"{model_name}: scoring split {best_split} ({len(validation_rows):,} rows)")
    scorer = get_scorer(scoring[rank_metric])
    score = scorer(pipeline, features.iloc[validation_rows], labels.iloc[validation_rows])

    if use_smote:
        folder_path = PATH_FOLDER_TRAINED_MODEL_WITH_SMOTE
    else:
        folder_path = PATH_FOLDER_TRAINED_MODEL_WOUT_SMOTE
    os.makedirs(folder_path, exist_ok=True)
    model_path = os.path.join(folder_path, f"{model_name}.pkl")

    print_status(f"{model_name}: saving the split {best_split} model to {model_path}")
    dump(pipeline, model_path)
    end_status(
        f"{model_name}: saved the split {best_split} model ({pipeline.named_steps['pca'].n_components_} PCs) "
        f"to {model_path}, {rank_metric} {score:.4f} (search {split_scores[best_split]:.4f})"
    )
```

Kode Program 5.62 Pembentukan dan penyimpanan model akhir dari pembagian terbaik
