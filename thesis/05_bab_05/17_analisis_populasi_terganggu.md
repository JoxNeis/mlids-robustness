## **5.17 Implementasi Analisis Populasi Baris Utuh dan Baris Terganggu**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.17, yaitu analisis yang memisahkan baris utuh dari baris terganggu pada hasil pengujian ketahanan. Kode program disusun mengikuti Langkah 1 hingga 7 pada algoritma di Subbab 4.17, dilanjutkan dengan kode untuk menyajikan hasil analisis. Analisis dilaksanakan pada prediksi yang telah tersimpan sehingga tidak memerlukan pelatihan maupun prediksi ulang.

Kode Program 5.139 mendefinisikan folder hasil analisis (*PATH_FOLDER_ANALYSIS*), lokasi berkas penanda baris terganggu (*PATH_COVERAGE_MASKS*), nama kedua populasi (*NOISE_POPULATIONS*, yaitu *intact* dan *corrupted*), dan jumlah komponen yang dibandingkan pada pemulihan penanda (*MASK_PROBE_COMPONENTS*, yaitu 4).

```python
PATH_FOLDER_ANALYSIS = "analysis"
PATH_COVERAGE_MASKS = os.path.join(PATH_FOLDER_ANALYSIS, "coverage-masks.npz")

NOISE_POPULATIONS = ("intact", "corrupted")

MASK_PROBE_COMPONENTS = 4
```

Kode Program 5.139 Pengaturan analisis populasi

Kode Program 5.140 mengimplementasikan Langkah 2 dan 3. Fungsi *recover_coverage_mask* membandingkan empat komponen utama pertama pada data bersih dan data ternoise, berkas demi berkas, dengan *numpy.isclose* menggunakan toleransi relatif 10⁻⁷ dan toleransi mutlak 10⁻¹⁰ serta *NaN* dianggap sama. Baris yang berbeda pada sedikitnya satu komponen ditandai terganggu. Fungsi *load_coverage_masks* memuat penanda seluruh skenario dari berkas *cache* apabila lengkap, dan apabila belum, memulihkan penanda kemudian menyimpannya dengan *numpy.savez_compressed*. Sel kedua memeriksa, dengan pernyataan *assert*, bahwa proporsi baris terganggu pada setiap skenario berbeda dari intensitasnya kurang dari 10⁻⁴.

```python
def recover_coverage_mask(
    scenario: str,
    features_folder_name: str = PATH_FOLDER_IPCA,
    noise_folder_name: str = PATH_FOLDER_NOISE_IPCA,
    split_name: str = NOISE_SPLIT,
    n_components: int = MASK_PROBE_COMPONENTS,
) -> np.ndarray:
    clean_paths = get_split_file_paths(features_folder_name, split_name)
    columns = pl.read_parquet(clean_paths[0]).columns[:n_components]
    noise_split_folder = get_split_folder(
        get_scenario_folder(noise_folder_name, scenario), split_name
    )

    parts = []
    for clean_path in clean_paths:
        clean = pl.read_parquet(clean_path, columns=columns).to_numpy()
        noisy = pl.read_parquet(
            os.path.join(noise_split_folder, os.path.basename(clean_path)), columns=columns
        ).to_numpy()
        parts.append((~np.isclose(noisy, clean, rtol=1e-7, atol=1e-10, equal_nan=True)).any(axis=1))

    return np.concatenate(parts)


def load_coverage_masks(
    scenarios: list[tuple[str, float, str]] | None = None,
    cache_file: str = PATH_COVERAGE_MASKS,
) -> dict[str, np.ndarray]:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    names = [scenario for _, _, scenario in scenarios]

    if os.path.exists(cache_file):
        stored = np.load(cache_file)
        if all(name in stored.files for name in names):
            print(f"Read {len(names)} coverage masks from {cache_file}.")
            return {name: stored[name] for name in names}

    masks = {}
    for name in names:
        with step(f"recovering the coverage mask of '{name}'"):
            masks[name] = recover_coverage_mask(name)
        print(f"    {masks[name].mean() * 100:.3f}% of records corrupted")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, **masks)
    print(f"Saved {len(masks)} coverage masks to {cache_file}.")
    return masks

coverage_masks = load_coverage_masks()

for _, intensity, scenario in list_noise_scenarios():
    share = float(coverage_masks[scenario].mean())
    assert abs(share - intensity) < 1e-4, f"{scenario}: {share:.5f} corrupted, expected {intensity}"

print(f"All {len(coverage_masks)} masks match the coverage their scenario name claims.")
```

Kode Program 5.140 Pemulihan dan verifikasi penanda baris terganggu

Fungsi *audit_noise_baselines* pada Kode Program 5.141 mengimplementasikan Langkah 4. Untuk setiap model dan skenario, fungsi ini memuat prediksi data bersih dan prediksi skenario, kemudian menghitung proporsi baris utuh yang diprediksi berbeda (*disagreement*), akurasi prediksi data bersih tersimpan (*stored_clean*), dan akurasi prediksi skenario pada baris utuh (*recovered_clean*). Sel kedua menjalankannya dan meringkasnya untuk setiap model.

```python
def audit_noise_baselines(
    true_labels=None,
    scenarios: list[tuple[str, float, str]] | None = None,
    models: list[str] | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    true = to_labels(test_y if true_labels is None else true_labels)
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    models = list(MODEL_SUBFOLDERS) if models is None else models
    masks = coverage_masks if masks is None else masks

    rows = []
    for model_name in models:
        subfolder = MODEL_SUBFOLDERS[model_name]
        try:
            baseline = load_prediction_chunks(
                get_prediction_folder(subfolder, get_noise_split_name(NOISE_BASELINE))
            )
        except FileNotFoundError:
            print(f"{model_name}: no cached clean predictions, skipped.")
            continue

        for _, _, scenario in scenarios:
            folder = get_prediction_folder(subfolder, get_noise_split_name(scenario))
            try:
                pred = load_prediction_chunks(folder)
            except FileNotFoundError:
                continue

            intact = ~masks[scenario]
            rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "disagreement": float((baseline[intact] != pred[intact]).mean()),
                    "stored_clean": float((baseline == true).mean()),
                    "recovered_clean": float((pred[intact] == true[intact]).mean()),
                }
            )

    return pd.DataFrame(rows)

noise_baseline_audit = audit_noise_baselines()
noise_baseline_audit.groupby("model", sort=False).agg(
    disagreement=("disagreement", "mean"),
    stored_clean=("stored_clean", "first"),
    recovered_clean=("recovered_clean", "mean"),
).round(4)
```

Kode Program 5.141 Audit skor acuan data bersih

Kode Program 5.142 mengimplementasikan Langkah 5. Fungsi *score_noise_populations* menghitung metrik pada baris utuh dan baris terganggu untuk setiap model dan skenario terhadap himpunan kelas yang sama, beserta jumlah barisnya. Sel kedua menjalankannya dan menyimpan hasilnya ke *hyperparameter-search/noise-populations.csv*. Fungsi *pivot_noise_populations* meringkas F1-*score* menurut populasi dan jenis gangguan, dan sel terakhir menjalankannya.

```python
def score_noise_populations(
    true_labels=None,
    scenarios: list[tuple[str, float, str]] | None = None,
    models: list[str] | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    true = to_labels(test_y if true_labels is None else true_labels)
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    models = list(MODEL_SUBFOLDERS) if models is None else models
    masks = coverage_masks if masks is None else masks
    labels = get_noise_labels()

    rows = []
    for noise_type, intensity, scenario in scenarios:
        for model_name in models:
            folder = get_prediction_folder(
                MODEL_SUBFOLDERS[model_name], get_noise_split_name(scenario)
            )
            try:
                pred = load_prediction_chunks(folder)
            except FileNotFoundError:
                print(f"{model_name} / {scenario}: no cached predictions, skipped.")
                continue

            corrupted = masks[scenario]
            for population, selected in (("intact", ~corrupted), ("corrupted", corrupted)):
                scores = evaluate(pred=pred[selected], true=true[selected], labels=labels).iloc[0]
                rows.append(
                    {
                        "model": model_name,
                        "noise": noise_type,
                        "intensity": intensity,
                        "scenario": scenario,
                        "population": population,
                        **{name: float(value) for name, value in scores.items()},
                        "rows": int(selected.sum()),
                    }
                )

    return pd.DataFrame(rows)

noise_populations = score_noise_populations()
save_search_results(noise_populations, "noise-populations.csv", PATH_FOLDER_SEARCH_RESULT)
noise_populations.head(10)

def pivot_noise_populations(
    populations: pd.DataFrame, score_column: str = SCORE_COLUMN
) -> pd.DataFrame:
    table = populations.pivot_table(
        index="model", columns=["population", "noise"], values=score_column
    )
    return table.reindex(
        index=[name for name in MODEL_SUBFOLDERS if name in table.index],
        columns=pd.MultiIndex.from_product([list(NOISE_POPULATIONS), list(NOISE_TYPES)]),
    )

pivot_noise_populations(noise_populations).round(4)
```

Kode Program 5.142 Penilaian baris utuh dan baris terganggu secara terpisah

Fungsi *measure_mixture_residual* pada Kode Program 5.143 mengimplementasikan Langkah 6. Fungsi ini membandingkan akurasi gabungan yang tersimpan pada berkas skor dengan campuran akurasi baris utuh dan baris terganggu yang berbobot (1 − *p*) dan *p* (Persamaan 4.11), dan mengembalikan selisihnya sebagai residu. Sel kedua menjalankannya dan mencetak residu mutlak terbesar.

```python
def measure_mixture_residual(
    populations: pd.DataFrame, folder_name: str = PATH_FOLDER_NOISE_SCORES
) -> pd.DataFrame:
    rows = []
    for (model_name, scenario), group in populations.groupby(["model", "scenario"], sort=False):
        stored = load_noise_score(model_name, scenario, folder_name)
        if stored is None:
            continue

        indexed = group.set_index("population")
        share = float(indexed["intensity"].iloc[0])
        mixed = (
            (1.0 - share) * float(indexed.loc["intact", "accuracy"])
            + share * float(indexed.loc["corrupted", "accuracy"])
        )
        rows.append(
            {
                "model": model_name,
                "scenario": scenario,
                "scored": float(stored["accuracy"]),
                "mixed": mixed,
                "residual": float(stored["accuracy"]) - mixed,
            }
        )

    return pd.DataFrame(rows)

noise_mixture = measure_mixture_residual(noise_populations)
print(f"largest residual over {len(noise_mixture)} passes: {noise_mixture['residual'].abs().max():.3e}")
noise_mixture.head(10)
```

Kode Program 5.143 Verifikasi identitas campuran akurasi

Fungsi *compare_population_per_class* pada Kode Program 5.144 mengimplementasikan bagian pertama Langkah 7. Fungsi ini menyusun tabel metrik per kelas, dengan F1-*score* sebagai bawaan, untuk setiap model pada baris utuh atau baris terganggu suatu skenario. Sel kedua menjalankannya pada baris terganggu skenario *gaussian-30*.

```python
def compare_population_per_class(
    scenario: str,
    population: str,
    true_labels=None,
    masks: dict[str, np.ndarray] | None = None,
    subfolders: dict[str, str] | None = None,
    class_names: list[str] | None = None,
    metric: str = "f1",
) -> pd.DataFrame:
    true = to_labels(test_y if true_labels is None else true_labels)
    masks = coverage_masks if masks is None else masks
    subfolders = MODEL_SUBFOLDERS if subfolders is None else subfolders
    class_names = CLASS_NAMES if class_names is None else class_names

    corrupted = masks[scenario]
    selected = corrupted if population == "corrupted" else ~corrupted

    columns = {}
    for model_name, subfolder in subfolders.items():
        folder = get_prediction_folder(subfolder, get_noise_split_name(scenario))
        try:
            pred = load_prediction_chunks(folder)
        except FileNotFoundError:
            print(f"{model_name}: no cached predictions for {scenario}, skipped.")
            continue

        report = get_classification_report(
            pred=pred[selected], true=true[selected], class_names=class_names
        )
        columns[model_name] = report.set_index("class")[metric]

    return pd.DataFrame(columns).reindex(class_names)

compare_population_per_class("gaussian-30", "corrupted").round(3)
```

Kode Program 5.144 F1-score per kelas pada satu populasi

Kode Program 5.145 mengimplementasikan bagian kedua Langkah 7. Fungsi *build_conditional_confusion* menyusun *confusion matrix* suatu model dan skenario pada baris utuh atau baris terganggu. Fungsi *measure_error_composition* menjumlahkan matriks tersebut pada ketiga intensitas setiap jenis gangguan, kemudian menghitung laju kesalahan dan proporsi kesalahan berupa serangan yang diprediksi sebagai *Benign*, serangan yang diprediksi sebagai serangan lain, dan *Benign* yang diprediksi sebagai serangan. Sel kedua menjalankannya dan menyimpan hasilnya ke *hyperparameter-search/noise-error-composition.csv*.

```python
def build_conditional_confusion(
    model_name: str,
    scenario: str,
    population: str,
    true_labels=None,
    masks: dict[str, np.ndarray] | None = None,
    labels: list[int] | None = None,
) -> np.ndarray:
    true = to_labels(test_y if true_labels is None else true_labels)
    masks = coverage_masks if masks is None else masks
    labels = get_noise_labels() if labels is None else labels

    pred = load_prediction_chunks(
        get_prediction_folder(MODEL_SUBFOLDERS[model_name], get_noise_split_name(scenario))
    )
    corrupted = masks[scenario]
    selected = corrupted if population == "corrupted" else ~corrupted

    return confusion_matrix(true[selected], pred[selected], labels=labels)


def measure_error_composition(
    population: str = "corrupted",
    scenarios: list[tuple[str, float, str]] | None = None,
    models: list[str] | None = None,
    benign_label: int = 0,
    **kwargs,
) -> pd.DataFrame:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    models = list(MODEL_SUBFOLDERS) if models is None else models

    rows = []
    for model_name in models:
        for noise_type in NOISE_TYPES:
            matrices = [
                build_conditional_confusion(model_name, scenario, population, **kwargs)
                for kind, _, scenario in scenarios
                if kind == noise_type
            ]
            if not matrices:
                continue

            matrix = np.sum(matrices, axis=0).astype(float)
            attack = [i for i in range(matrix.shape[0]) if i != benign_label]
            errors = matrix.sum() - np.trace(matrix)
            if errors == 0:
                continue

            attack_block = matrix[np.ix_(attack, attack)]
            rows.append(
                {
                    "model": model_name,
                    "noise": noise_type,
                    "error_rate": float(errors / matrix.sum()),
                    "attack_to_benign": float(matrix[attack, benign_label].sum() / errors),
                    "attack_to_attack": float((attack_block.sum() - np.trace(attack_block)) / errors),
                    "benign_to_attack": float(matrix[benign_label, attack].sum() / errors),
                }
            )

    return pd.DataFrame(rows)

noise_error_composition = measure_error_composition()
save_search_results(noise_error_composition, "noise-error-composition.csv", PATH_FOLDER_SEARCH_RESULT)
noise_error_composition.round(4)
```

Kode Program 5.145 Matriks konfusi bersyarat dan komposisi kesalahan

Kode Program 5.146 menetapkan warna kedua populasi dan gaya dasar grafik pada fungsi *_dress*. Fungsi *plot_baseline_audit* menggambar akurasi data bersih tersimpan dibandingkan akurasi yang dipulihkan, serta proporsi baris utuh yang diprediksi berbeda, untuk setiap model. Sel terakhir menjalankannya.

```python
NOISE_POPULATION_COLOURS = {"intact": "#2a78d6", "corrupted": "#eb6834"}

NOISE_SURFACE, NOISE_GRID = "#fcfcfb", "#e8e7e3"
NOISE_INK, NOISE_MUTED = "#0b0b0b", "#52514e"


def _dress(axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    axes.set_facecolor(NOISE_SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(NOISE_GRID)
    axes.tick_params(colors=NOISE_MUTED, length=0, labelsize=8)
    axes.grid(color=NOISE_GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    if title:
        axes.set_title(title, color=NOISE_INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        axes.set_xlabel(xlabel, color=NOISE_MUTED, fontsize=9)
    if ylabel:
        axes.set_ylabel(ylabel, color=NOISE_MUTED, fontsize=9)

def plot_baseline_audit(audit: pd.DataFrame) -> None:
    summary = audit.groupby("model", sort=False).agg(
        stored=("stored_clean", "first"),
        recovered=("recovered_clean", "mean"),
        disagreement=("disagreement", "mean"),
    )
    summary = summary.reindex([name for name in MODEL_SUBFOLDERS if name in summary.index])
    models = list(summary.index)
    positions = np.arange(len(models))
    width = 0.36

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    figure.patch.set_facecolor(NOISE_SURFACE)

    for offset, column, population in (
        (-width / 2 - 0.01, "stored", "intact"),
        (+width / 2 + 0.01, "recovered", "corrupted"),
    ):
        values = summary[column].to_numpy(dtype=float)
        left.bar(
            positions + offset, values, width,
            color=NOISE_POPULATION_COLOURS[population], zorder=3,
            label="stored test/ baseline" if column == "stored" else "recovered (model on disk)",
        )
        for position, value in zip(positions + offset, values):
            left.text(position, value + 0.008, f"{value:.3f}", ha="center", fontsize=7.5, color=NOISE_MUTED)

    benign_share = float((to_labels(test_y) == 0).mean())
    left.axhline(benign_share, color=NOISE_MUTED, ls=(0, (4, 3)), lw=1.1, zorder=2)
    left.text(-0.45, benign_share - 0.016, f"all-Benign = {benign_share:.3f}",
              ha="left", fontsize=7.5, color=NOISE_MUTED)
    left.set_xticks(positions)
    left.set_xticklabels(models, rotation=18, ha="right")
    left.set_ylim(0.75, 1.06)
    left.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    _dress(left, "a. Clean accuracy: stored vs actual", ylabel="accuracy")

    shares = summary["disagreement"].to_numpy(dtype=float) * 100
    right.barh(positions, shares, 0.55,
               color=[NOISE_MODEL_COLOURS.get(name, "#2a78d6") for name in models], zorder=3)
    for position, value in zip(positions, shares):
        right.text(value + 0.25, position, f"{value:.2f}%", va="center", fontsize=8, color=NOISE_MUTED)
    right.set_yticks(positions)
    right.set_yticklabels(models)
    right.invert_yaxis()
    right.set_xlim(0, max(shares.max(), 1.0) * 1.25)
    _dress(right, "b. Disagreement on records that were never corrupted",
           xlabel="share of intact records predicted differently (%)")
    right.text(0.99, -0.20,
               "these records' features are bit-identical to the clean split,\n"
               "so a correctly-paired baseline would read 0.00%",
               transform=right.transAxes, ha="right", va="top",
               fontsize=7.5, color=NOISE_MUTED, style="italic")

    figure.tight_layout()
    plt.show()

plot_baseline_audit(noise_baseline_audit)
```

Kode Program 5.146 Visualisasi audit skor acuan data bersih

Fungsi *plot_mixture_parity* pada Kode Program 5.147 menggambar akurasi gabungan yang tersimpan terhadap campuran akurasi kedua populasi untuk setiap pengujian, beserta garis kesamaan dan residu mutlak terbesar. Sel kedua menjalankannya.

```python
def plot_mixture_parity(mixture: pd.DataFrame) -> None:
    figure, axes = plt.subplots(figsize=(5.6, 5.0))
    figure.patch.set_facecolor(NOISE_SURFACE)

    low = float(min(mixture["mixed"].min(), mixture["scored"].min())) - 0.01
    high = float(max(mixture["mixed"].max(), mixture["scored"].max())) + 0.01
    axes.plot([low, high], [low, high], color=NOISE_MUTED, ls=(0, (4, 3)), lw=1.1, zorder=2)

    for model_name in MODEL_SUBFOLDERS:
        rows = mixture[mixture["model"] == model_name]
        if rows.empty:
            continue
        axes.scatter(
            rows["mixed"], rows["scored"], s=46,
            color=NOISE_MODEL_COLOURS.get(model_name, "#2a78d6"),
            marker=NOISE_MODEL_MARKERS.get(model_name, "o"),
            edgecolor=NOISE_SURFACE, linewidth=1.2, label=model_name, zorder=3,
        )

    axes.text(0.04, 0.95, f"largest residual over all {len(mixture)} passes: "
                          f"{mixture['residual'].abs().max():.1e}",
              transform=axes.transAxes, fontsize=8, color=NOISE_MUTED)
    axes.legend(frameon=False, fontsize=8, loc="lower right")
    _dress(axes, "Pooled accuracy is the mixture of the two populations",
           xlabel=r"$(1-p)\cdot$acc$_{intact} + p\cdot$acc$_{corrupted}$",
           ylabel="pooled accuracy as scored in 19.8")

    figure.tight_layout()
    plt.show()

plot_mixture_parity(noise_mixture)
```

Kode Program 5.147 Visualisasi kesesuaian identitas campuran

Fungsi *plot_noise_populations* pada Kode Program 5.148 menggambar F1-*score* baris utuh dan baris terganggu untuk setiap model pada setiap jenis gangguan sebagai diagram batang berdampingan. Sel kedua menjalankannya.

```python
def plot_noise_populations(
    populations: pd.DataFrame, score_column: str = SCORE_COLUMN
) -> None:
    models = [name for name in MODEL_SUBFOLDERS if name in set(populations["model"])]
    short = {"Random Forest": "RF", "XGBoost": "XGB"}
    positions = np.arange(len(models))
    width = 0.36

    figure, axes_row = plt.subplots(1, len(NOISE_TYPES), figsize=(3.7 * len(NOISE_TYPES), 4.2), sharey=True)
    figure.patch.set_facecolor(NOISE_SURFACE)

    for axes, noise_type in zip(np.atleast_1d(axes_row), NOISE_TYPES):
        panel = populations[populations["noise"] == noise_type]
        for offset, population in ((-width / 2 - 0.01, "intact"), (+width / 2 + 0.01, "corrupted")):
            values = [
                panel[(panel["model"] == name) & (panel["population"] == population)][score_column].mean()
                for name in models
            ]
            axes.bar(
                positions + offset, values, width, zorder=3,
                color=NOISE_POPULATION_COLOURS[population],
                label=population if noise_type == NOISE_TYPES[0] else None,
            )
        axes.set_xticks(positions)
        axes.set_xticklabels([short.get(name, name) for name in models], fontsize=8.5)
        _dress(axes, noise_type, ylabel=score_column if noise_type == NOISE_TYPES[0] else None)

    np.atleast_1d(axes_row)[0].legend(frameon=False, fontsize=8, loc="upper right")
    figure.suptitle(
        f"{score_column} on uncorrupted vs corrupted records of the same run",
        x=0.5, y=1.02, fontsize=10.5, color=NOISE_INK,
    )
    figure.tight_layout()
    plt.show()

plot_noise_populations(noise_populations)
```

Kode Program 5.148 Visualisasi skor baris utuh dan baris terganggu

Fungsi *plot_population_confusion* pada Kode Program 5.149 menggambar *confusion matrix* yang dinormalisasi menurut kelas sebenarnya, berdampingan untuk baris utuh dan baris terganggu, pada satu pasangan model dan skenario. Sel kedua menjalankannya pada model *KNN* dan skenario *gaussian-30*.

```python
def plot_population_confusion(
    model_name: str,
    scenario: str,
    class_names: list[str] | None = None,
    **kwargs,
) -> None:
    class_names = CLASS_NAMES if class_names is None else class_names
    ramp = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    colours = matplotlib.colors.LinearSegmentedColormap.from_list("confusion", ramp)

    figure, axes_row = plt.subplots(1, 2, figsize=(13, 5.9))
    figure.patch.set_facecolor(NOISE_SURFACE)

    for axes, population in zip(axes_row, NOISE_POPULATIONS):
        matrix = build_conditional_confusion(model_name, scenario, population, **kwargs).astype(float)
        shares = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = axes.imshow(shares, cmap=colours, vmin=0.0, vmax=1.0, aspect="auto")

        for row in range(shares.shape[0]):
            for column in range(shares.shape[1]):
                value = shares[row, column]
                if value < 0.005:
                    continue
                axes.text(
                    column, row, "1.0" if value >= 0.995 else f"{value:.2f}"[1:],
                    ha="center", va="center", fontsize=6,
                    color="#ffffff" if value > 0.55 else NOISE_INK,
                )

        axes.set_xticks(range(len(class_names)))
        axes.set_xticklabels(class_names, rotation=90, fontsize=7)
        axes.set_yticks(range(len(class_names)))
        axes.set_yticklabels(class_names if population == "intact" else [], fontsize=7)
        axes.grid(False)
        axes.tick_params(colors=NOISE_MUTED, length=0)
        for side in ("top", "right", "bottom", "left"):
            axes.spines[side].set_visible(False)
        axes.set_title(f"{model_name}, {scenario} — {population} records",
                       color=NOISE_INK, fontsize=10, loc="left", pad=8)
        axes.set_xlabel("predicted", color=NOISE_MUTED, fontsize=9)
        if population == "intact":
            axes.set_ylabel("true", color=NOISE_MUTED, fontsize=9)

    figure.colorbar(image, ax=axes_row, fraction=0.02, pad=0.02,
                    label="share of the true class's records")
    plt.show()

plot_population_confusion("KNN", "gaussian-30")
```

Kode Program 5.149 Visualisasi matriks konfusi pada kedua populasi
