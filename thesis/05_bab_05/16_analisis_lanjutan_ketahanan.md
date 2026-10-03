## **5.16 Implementasi Analisis Lanjutan Ketahanan Model**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.16, yaitu analisis lanjutan yang memanfaatkan tabel hasil, metrik, dan penurunan kinerja dari Subbab 5.15 tanpa pelatihan maupun prediksi ulang. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.16.

Kode Program 5.106 mengimplementasikan bagian pertama Langkah 1. Sel tersebut memilih metrik populasi baris utuh pada seluruh skenario gangguan, menghitung laju perubahan prediksi terbesar untuk setiap model, dan menampilkan nilai terbesar dari seluruh model, yang harus bernilai nol. Tabel F1-*score* makro pada populasi baris utuh dan baris terganggu untuk setiap jenis gangguan juga ditampilkan.

```python
is_intact = (metrics[POPULATION_COLUMN] == "intact") & (metrics[SCENARIO_COLUMN] != CLEAN_SCENARIO)
intact_flip_rate = metrics[is_intact].groupby([SMOTE_SETTING_COLUMN, MODEL_COLUMN], sort=False)["flip_rate"].max()
end_status(f"At most {intact_flip_rate.max():.4%} of the intact rows changed prediction from the clean test split")
display(intact_flip_rate.rename("max_intact_flip_rate").to_frame())
display(
    metrics[metrics[POPULATION_COLUMN] != "all"].pivot_table(
        index=[SMOTE_SETTING_COLUMN, MODEL_COLUMN],
        columns=[NOISE_TYPE_COLUMN, POPULATION_COLUMN],
        values=ROBUSTNESS_METRIC,
    ).round(4)
)
```

Kode Program 5.106 Pemeriksaan laju perubahan prediksi pada baris utuh

Fungsi *check_mixture_identity* pada Kode Program 5.107 mengimplementasikan bagian kedua Langkah 1. Fungsi ini menyusun akurasi dan jumlah baris ketiga populasi pada setiap pengujian skenario gangguan, menghitung proporsi baris terganggu, menghitung campuran akurasi baris utuh dan baris terganggu sesuai ruas kanan Persamaan 4.18, dan menghitung residunya terhadap akurasi seluruh baris. Sel kedua menyimpan hasilnya pada file *mixture-check.csv* dan menampilkan residu mutlak terbesar.

```python
def check_mixture_identity(metrics: pd.DataFrame) -> pd.DataFrame:
    noise_metrics = metrics[metrics[SCENARIO_COLUMN] != CLEAN_SCENARIO]
    table = noise_metrics.pivot_table(
        index=[SMOTE_SETTING_COLUMN, MODEL_COLUMN, SCENARIO_COLUMN, INTENSITY_COLUMN],
        columns=POPULATION_COLUMN,
        values=["accuracy", ROW_COUNT_COLUMN],
    )
    noisy_fraction = table[(ROW_COUNT_COLUMN, "noisy")] / table[(ROW_COUNT_COLUMN, "all")]
    mixture_check = pd.DataFrame({
        "noisy_fraction": noisy_fraction,
        "accuracy": table[("accuracy", "all")],
        "intact_accuracy": table[("accuracy", "intact")],
        "noisy_accuracy": table[("accuracy", "noisy")],
    })
    mixture_check["mixed_accuracy"] = (
        (1 - noisy_fraction) * mixture_check["intact_accuracy"] + noisy_fraction * mixture_check["noisy_accuracy"]
    )
    mixture_check["residual"] = mixture_check["accuracy"] - mixture_check["mixed_accuracy"]
    return mixture_check.reset_index()

mixture_check = check_mixture_identity(metrics)
mixture_check.to_csv(os.path.join(PATH_FOLDER_EVALUATION, MIXTURE_CHECK_FILE_NAME), index=False)
largest_residual = mixture_check["residual"].abs().max()
end_status(f"The accuracy differs from the mixture of the intact and noisy rows by at most {largest_residual:.2e}")
display(mixture_check.round(6))
```

Kode Program 5.107 Pemeriksaan identitas campuran akurasi

Fungsi *measure_smote_effect* pada Kode Program 5.108 mengimplementasikan Langkah 2. Fungsi ini memisahkan tabel penurunan kinerja menurut pengaturan SMOTE, memasangkan kedua pengaturan berdasarkan model, skenario, jenis gangguan, intensitas, populasi, dan metrik, kemudian menghitung selisih nilai dan selisih retensi dengan SMOTE dikurangi tanpa SMOTE. Sel kedua menyimpan hasilnya pada file *smote-effect.csv* dan, apabila kedua pengaturan telah diuji, menggambar selisih F1-*score* makro pada seluruh baris sebagai peta panas dengan skala warna divergen yang simetris terhadap nol.

```python
def measure_smote_effect(degradation: pd.DataFrame) -> pd.DataFrame:
    keys = [MODEL_COLUMN, SCENARIO_COLUMN, NOISE_TYPE_COLUMN, INTENSITY_COLUMN, POPULATION_COLUMN, METRIC_COLUMN]
    values = [VALUE_COLUMN, RETENTION_COLUMN]
    with_smote = degradation[degradation[SMOTE_SETTING_COLUMN] == "with-smote"].set_index(keys)[values]
    wout_smote = degradation[degradation[SMOTE_SETTING_COLUMN] == "wout-smote"].set_index(keys)[values]
    smote_effect = with_smote.join(wout_smote, how="inner", lsuffix="_with_smote", rsuffix="_wout_smote")
    for column in values:
        difference = smote_effect[f"{column}_with_smote"] - smote_effect[f"{column}_wout_smote"]
        smote_effect[f"{column}_difference"] = difference
    return smote_effect.reset_index()

smote_effect = measure_smote_effect(degradation)
smote_effect.to_csv(os.path.join(PATH_FOLDER_EVALUATION, SMOTE_EFFECT_FILE_NAME), index=False)
if smote_effect.empty:
    end_status("Both SMOTE settings need predictions to measure the effect of SMOTE")
else:
    is_shown = (smote_effect[METRIC_COLUMN] == ROBUSTNESS_METRIC) & (smote_effect[POPULATION_COLUMN] == "all")
    difference = smote_effect[is_shown].pivot(index=MODEL_COLUMN, columns=SCENARIO_COLUMN, values="value_difference")
    difference = difference.reindex(index=list(EVALUATED_MODELS), columns=list_scenario_names())
    difference = difference.rename(index=EVALUATED_MODELS)
    largest_difference = np.nanmax(np.abs(difference.to_numpy()))
    display(difference.round(4))
    plot_heatmap(
        difference,
        f"{METRIC_LABELS[ROBUSTNESS_METRIC]} With SMOTE Minus Without SMOTE",
        colormap=DIVERGING_COLORMAP,
        value_range=(-largest_difference, largest_difference),
        annotate=True,
        annotation_format=".3f",
        xlabel="Scenario",
        ylabel="Model",
        colorbar_label="With SMOTE - without SMOTE",
        figsize=(14, 6),
        save_path=os.path.join(PATH_FOLDER_EVALUATION, f"{ROBUSTNESS_METRIC}-smote-effect.png"),
    )
```

Kode Program 5.108 Pengukuran efek SMOTE

Kode Program 5.109 mengimplementasikan Langkah 3. Fungsi *score_classes* menghitung presisi, *recall*, dan F1-*score* setiap kelas untuk prediksi pada skenario dan prediksi bersih menggunakan jumlah baris sebagai bobot sampel, dengan *zero_division=np.nan* sehingga kelas yang tidak pernah diprediksi tidak diberi skor nol. Untuk setiap kelas dihitung pula jumlah baris, jumlah prediksi, laju perubahan prediksi, proporsi yang diprediksi *Benign* pada skenario dan pada data bersih, serta rata-rata tingkat kepercayaan. Fungsi *measure_every_class* menjalankan *score_classes* pada setiap pengujian dan populasi, kemudian menghitung penurunan F1-*score* setiap kelas. Sel terakhir menyimpan hasilnya pada file *class-metrics.csv*.

```python
def score_classes(outcomes: pd.DataFrame, class_names: list[str], benign: int) -> pd.DataFrame:
    labels = outcomes[LABEL_COLUMN].to_numpy()
    predictions = outcomes[PREDICTION_COLUMN].to_numpy()
    clean_predictions = outcomes[CLEAN_PREDICTION_COLUMN].to_numpy()
    rows = outcomes[ROW_COUNT_COLUMN].to_numpy()
    confidence_sums = outcomes[CONFIDENCE_SUM_COLUMN].to_numpy()
    class_codes = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_codes, sample_weight=rows, zero_division=np.nan
    )
    clean_precision, clean_recall, clean_f1, _ = precision_recall_fscore_support(
        labels, clean_predictions, labels=class_codes, sample_weight=rows, zero_division=np.nan
    )

    records = []
    for class_code in class_codes:
        is_class = labels == class_code
        records.append({
            CLASS_COLUMN: class_names[class_code],
            "support": int(support[class_code]),
            "predicted": int(rows[predictions == class_code].sum()),
            "precision": precision[class_code],
            "recall": recall[class_code],
            "f1": f1[class_code],
            "clean_precision": clean_precision[class_code],
            "clean_recall": clean_recall[class_code],
            "clean_f1": clean_f1[class_code],
            "flip_rate": mean_per_row(rows * (predictions != clean_predictions), rows, is_class),
            "benign_rate": mean_per_row(rows * (predictions == benign), rows, is_class),
            "clean_benign_rate": mean_per_row(rows * (clean_predictions == benign), rows, is_class),
            "mean_confidence": mean_per_row(confidence_sums, rows, is_class),
        })
    return pd.DataFrame(records)

def measure_every_class(outcomes: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    benign = class_names.index(BENIGN_CLASS)
    groups = outcomes.groupby(COMBINATION_COLUMNS, sort=False)
    tables = []
    started = time.time()
    for group_index, (combination, combination_outcomes) in enumerate(groups, start=1):
        smote_setting, model_name, scenario, _, _ = combination
        for population in POPULATIONS:
            population_outcomes = select_population(combination_outcomes, population)
            if population_outcomes.empty:
                continue
            table = score_classes(population_outcomes, class_names, benign)
            table.insert(0, POPULATION_COLUMN, population)
            tables.append(add_combination_columns(table, smote_setting, model_name, scenario))
        print_progress(group_index, groups.ngroups, "Scoring every class", started=started)

    class_metrics = pd.concat(tables, ignore_index=True)
    class_metrics["f1_drop"] = class_metrics["clean_f1"] - class_metrics["f1"]
    end_status(f"Scored {len(class_names)} classes on {len(tables)} populations")
    return class_metrics

class_metrics = measure_every_class(outcomes, class_names)
class_metrics.to_csv(os.path.join(PATH_FOLDER_EVALUATION, CLASS_METRICS_FILE_NAME), index=False)
display(class_metrics.round(4))
```

Kode Program 5.109 Metrik setiap kelas

Fungsi *plot_class_heatmap* pada Kode Program 5.110 menggambar sebuah metrik per kelas untuk setiap pasangan model dan pengaturan SMOTE pada satu skenario dan populasi sebagai peta panas, kemudian menyimpannya. Sel kedua menggambar F1-*score* per kelas pada data uji bersih, serta penurunan F1-*score* per kelas dan proporsi setiap kelas yang diprediksi *Benign* pada baris terganggu untuk skenario berintensitas 30% pada setiap jenis gangguan.

```python
def plot_class_heatmap(
    class_metrics: pd.DataFrame,
    class_names: list[str],
    scenario: str,
    population: str,
    value_column: str,
    title: str,
    colorbar_label: str,
    colormap: mcolors.Colormap = SEQUENTIAL_COLORMAP,
    value_range: tuple[float | None, float | None] = (0, 1),
) -> None:
    is_shown = (class_metrics[SCENARIO_COLUMN] == scenario) & (class_metrics[POPULATION_COLUMN] == population)
    plot_heatmap(
        pivot_models(class_metrics[is_shown], CLASS_COLUMN, value_column, class_names),
        title,
        colormap=colormap,
        value_range=value_range,
        annotate=True,
        annotation_format=".2f",
        xlabel="Model",
        ylabel="Class",
        colorbar_label=colorbar_label,
        figsize=(14, 9),
        save_path=os.path.join(PATH_FOLDER_EVALUATION, f"class-{value_column}-{scenario}-{population}.png"),
    )

plot_class_heatmap(
    class_metrics,
    class_names,
    CLEAN_SCENARIO,
    "all",
    "f1",
    "Per-Class F1-Score on the Clean Test Split",
    "F1-score",
)
highest_intensity = max(NOISE_INTENSITIES)
for noise_type in NOISE_MAGNITUDES:
    scenario = get_noise_scenario_name(noise_type, highest_intensity)
    plot_class_heatmap(
        class_metrics,
        class_names,
        scenario,
        "noisy",
        "f1_drop",
        f"Per-Class F1-Score Drop on the Noisy Rows of {scenario} (clean - noisy, same rows)",
        "F1-score drop",
        DIVERGING_COLORMAP,
        (-1, 1),
    )
    plot_class_heatmap(
        class_metrics,
        class_names,
        scenario,
        "noisy",
        "benign_rate",
        f"Share of Each Class Predicted as Benign on the Noisy Rows of {scenario}",
        "Predicted as Benign",
    )
```

Kode Program 5.110 Peta panas metrik setiap kelas

Kode Program 5.111 mengimplementasikan Langkah 4. Fungsi *measure_error_composition* memilih populasi seluruh baris pada data uji bersih dan populasi baris terganggu pada skenario gangguan, menghitung laju kesalahan, kemudian merata-ratakan laju kesalahan, proporsi ketiga kelompok kesalahan, laju deteksi serangan, dan laju alarm palsu menurut pengaturan SMOTE, model, dan jenis gangguan. Fungsi *plot_error_composition* menggambar diagram batang bertumpuk dengan tinggi setiap bagian sama dengan proporsi kelompok kesalahan dikalikan laju kesalahan, sehingga tinggi total batang sama dengan laju kesalahan. Sel terakhir menyimpan tabel dan gambar komposisi kesalahan.

```python
def measure_error_composition(metrics: pd.DataFrame) -> pd.DataFrame:
    is_clean = (metrics[SCENARIO_COLUMN] == CLEAN_SCENARIO) & (metrics[POPULATION_COLUMN] == "all")
    is_noisy = metrics[POPULATION_COLUMN] == "noisy"
    rows = metrics[is_clean | is_noisy].copy()
    rows["error_rate"] = 1 - rows["accuracy"]
    columns = [
        "error_rate",
        "missed_attack_share",
        "attack_confusion_share",
        "false_alarm_share",
        "detection_rate",
        "false_alarm_rate",
    ]
    error_composition = rows.groupby([SMOTE_SETTING_COLUMN, MODEL_COLUMN, NOISE_TYPE_COLUMN], sort=False)[columns].mean()
    return error_composition.reset_index()

def plot_error_composition(
    error_composition: pd.DataFrame,
    bar_width: float = 0.6,
    figsize: tuple[int, int] = (20, 9),
    save_path: str | None = None,
) -> None:
    error_types = {
        "missed_attack_share": "Attack predicted as Benign",
        "attack_confusion_share": "Attack predicted as another attack",
        "false_alarm_share": "Benign predicted as attack",
    }
    noise_types = [CLEAN_SCENARIO] + list(NOISE_MAGNITUDES)
    positions = np.arange(len(EVALUATED_MODELS))
    fig, axes = plt.subplots(
        len(SMOTE_SETTINGS),
        len(noise_types),
        figsize=figsize,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row_index, (smote_setting, smote_label) in enumerate(SMOTE_SETTINGS.items()):
        for column_index, noise_type in enumerate(noise_types):
            ax = axes[row_index, column_index]
            is_panel = (
                (error_composition[SMOTE_SETTING_COLUMN] == smote_setting)
                & (error_composition[NOISE_TYPE_COLUMN] == noise_type)
            )
            panel = error_composition[is_panel].set_index(MODEL_COLUMN).reindex(list(EVALUATED_MODELS))
            bottom = np.zeros(len(positions))
            for type_index, (error_type, error_label) in enumerate(error_types.items()):
                heights = (panel[error_type] * panel["error_rate"]).fillna(0).to_numpy()
                ax.bar(
                    positions,
                    heights,
                    bar_width,
                    bottom=bottom,
                    label=error_label,
                    color=SERIES_COLORS[type_index],
                    zorder=3,
                )
                bottom += heights
            ax.set_title(f"{noise_type.capitalize()} ({smote_label})")
            ax.set_xticks(positions)
            ax.set_xticklabels(list(EVALUATED_MODELS.values()), rotation=30, ha="right")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)

    for ax in axes[:, 0]:
        ax.set_ylabel("Error rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle("Error Composition on the Clean Test Split and on the Noisy Rows (mean over intensities)")

    if save_path is not None:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()

error_composition = measure_error_composition(metrics)
error_composition.to_csv(os.path.join(PATH_FOLDER_EVALUATION, ERROR_COMPOSITION_FILE_NAME), index=False)
display(error_composition.round(4))
plot_error_composition(
    error_composition,
    save_path=os.path.join(PATH_FOLDER_EVALUATION, ERROR_COMPOSITION_PLOT_FILE_NAME),
)
```

Kode Program 5.111 Komposisi kesalahan

Kode Program 5.112 mengimplementasikan bagian pertama Langkah 5. Sel pertama menggambar peta panas laju perubahan prediksi dan laju lolos serangan pada baris terganggu. Fungsi *plot_flip_rate_against_displacement* menggambar laju perubahan prediksi setiap desil perpindahan terhadap rata-rata perpindahan desil tersebut pada skala logaritmik, untuk setiap model pada skenario berintensitas 30%. Sel terakhir menampilkan tabel desil perpindahan dan menyimpan grafik tersebut.

```python
plot_scenario_heatmap(
    metrics,
    "flip_rate",
    "noisy",
    "Share of the Noisy Rows Whose Prediction Changed From the Clean Test Split",
    "Prediction flip rate",
    os.path.join(PATH_FOLDER_EVALUATION, "flip_rate-noisy-heatmap.png"),
    value_range=(0, 1),
)
plot_scenario_heatmap(
    metrics,
    "evasion_rate",
    "noisy",
    "Share of the Detected Attacks Predicted as Benign Once Noisy",
    "Attack evasion rate",
    os.path.join(PATH_FOLDER_EVALUATION, "evasion_rate-noisy-heatmap.png"),
    value_range=(0, 1),
)

def plot_flip_rate_against_displacement(
    displacement_bins: pd.DataFrame,
    intensity: float = max(NOISE_INTENSITIES),
    figsize: tuple[int, int] = (18, 9),
    save_path: str | None = None,
) -> None:
    noise_types = list(NOISE_MAGNITUDES)
    fig, axes = plt.subplots(
        len(SMOTE_SETTINGS),
        len(noise_types),
        figsize=figsize,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    legend_handles = {}
    for row_index, (smote_setting, smote_label) in enumerate(SMOTE_SETTINGS.items()):
        for column_index, noise_type in enumerate(noise_types):
            ax = axes[row_index, column_index]
            scenario = get_noise_scenario_name(noise_type, intensity)
            for model_index, (model_name, model_label) in enumerate(EVALUATED_MODELS.items()):
                is_line = (
                    (displacement_bins[SMOTE_SETTING_COLUMN] == smote_setting)
                    & (displacement_bins[MODEL_COLUMN] == model_name)
                    & (displacement_bins[SCENARIO_COLUMN] == scenario)
                )
                rows = displacement_bins[is_line].sort_values("mean_displacement")
                if rows.empty:
                    continue
                (line,) = ax.plot(
                    rows["mean_displacement"],
                    rows["flip_rate"],
                    color=SERIES_COLORS[model_index],
                    marker=SERIES_MARKERS[model_index],
                    markersize=6,
                    linewidth=2,
                    zorder=3,
                )
                legend_handles.setdefault(model_label, line)
            ax.set_xscale("log")
            ax.set_title(f"{scenario} ({smote_label})")
            ax.grid(color=GRID_COLOR, linewidth=0.8, zorder=0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Displacement in principal components")
    for ax in axes[:, 0]:
        ax.set_ylabel("Prediction flip rate")
    handles = []
    labels = []
    for model_label in EVALUATED_MODELS.values():
        if model_label in legend_handles:
            handles.append(legend_handles[model_label])
            labels.append(model_label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    fig.suptitle(f"Prediction Flip Rate per Displacement Decile of the Noisy Rows ({round(intensity * 100)}% noisy rows)")

    if save_path is not None:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()

display(displacement_bins.round(4))
plot_flip_rate_against_displacement(
    displacement_bins,
    save_path=os.path.join(PATH_FOLDER_EVALUATION, FLIP_RATE_PLOT_FILE_NAME),
)
```

Kode Program 5.112 Stabilitas prediksi dan hubungannya dengan perpindahan

Kode Program 5.113 mengimplementasikan bagian kedua Langkah 5. Apabila sedikitnya satu model menyediakan tingkat kepercayaan, sel tersebut menampilkan rata-rata tingkat kepercayaan, rata-rata tingkat kepercayaan pada kesalahan, kesenjangan kepercayaan, dan *log loss* pada populasi seluruh baris dan baris terganggu untuk setiap jenis gangguan, kemudian menggambar peta panas proporsi kesalahan yakin pada baris terganggu. Apabila tidak ada model yang menyediakan tingkat kepercayaan, pemberitahuan ditampilkan.

```python
if metrics["mean_confidence"].notna().any():
    display(
        metrics[metrics[POPULATION_COLUMN] != "intact"].pivot_table(
            index=[SMOTE_SETTING_COLUMN, MODEL_COLUMN],
            columns=[NOISE_TYPE_COLUMN, POPULATION_COLUMN],
            values=["mean_confidence", "mean_error_confidence", "confidence_gap", "log_loss"],
        ).round(4)
    )
    plot_scenario_heatmap(
        metrics,
        "confident_error_share",
        "noisy",
        f"Share of the Errors on the Noisy Rows Made With at Least {CONFIDENT_ERROR_THRESHOLD:.0%} Confidence",
        "Confident error share",
        os.path.join(PATH_FOLDER_EVALUATION, "confident_error_share-noisy-heatmap.png"),
        value_range=(0, 1),
    )
else:
    end_status("No model gave probabilities, so there is no confidence to compare")
```

Kode Program 5.113 Analisis tingkat kepercayaan model

Kode Program 5.114 mengimplementasikan Langkah 6. Fungsi *get_confusion_matrix* menyusun matriks konfusi lima belas kelas dari tabel hasil menggunakan *confusion_matrix* dengan jumlah baris sebagai bobot sampel, dengan pilihan kolom kelas sebenarnya dan kolom prediksi. Fungsi *plot_confusion_matrix* menggambar matriks tersebut sebagai peta panas berskala logaritmik beranotasi. Fungsi *plot_every_matrix* menggambar matriks konfusi pada populasi seluruh baris dan baris terganggu untuk setiap pengujian, serta matriks transisi prediksi bersih terhadap prediksi pada skenario untuk baris terganggu, kemudian menyimpan gambarnya pada folder *matrices* menurut pengaturan SMOTE dan model. Matriks pada data uji bersih ditampilkan langsung, sedangkan matriks skenario gangguan hanya disimpan kecuali *SHOW_NOISY_MATRICES* bernilai *True*. Sel terakhir menjalankannya apabila *SAVE_MATRIX_PLOTS* bernilai *True*.

```python
def get_confusion_matrix(
    outcomes: pd.DataFrame,
    class_names: list[str],
    true_column: str = LABEL_COLUMN,
    predicted_column: str = PREDICTION_COLUMN,
) -> pd.DataFrame:
    matrix = confusion_matrix(
        outcomes[true_column],
        outcomes[predicted_column],
        labels=np.arange(len(class_names)),
        sample_weight=outcomes[ROW_COUNT_COLUMN],
    )
    confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
    return confusion.rename_axis(index=true_column, columns=predicted_column)

def plot_confusion_matrix(
    confusion: pd.DataFrame,
    title: str,
    save_path: str | None = None,
    show: bool = True,
    xlabel: str = "Predicted Class",
    ylabel: str = "True Class",
) -> None:
    plot_heatmap(
        confusion,
        title,
        log_scale=True,
        annotate=True,
        xlabel=xlabel,
        ylabel=ylabel,
        colorbar_label="Row Count (log scale)",
        figsize=(14, 11),
        save_path=save_path,
        show=show,
    )

def plot_every_matrix(
    outcomes: pd.DataFrame,
    class_names: list[str],
    show_noisy: bool = SHOW_NOISY_MATRICES,
    matrices_folder_path: str = PATH_FOLDER_EVALUATION_MATRICES,
) -> None:
    groups = outcomes.groupby(COMBINATION_COLUMNS, sort=False)
    started = time.time()
    for group_index, (combination, combination_outcomes) in enumerate(groups, start=1):
        smote_setting, model_name, scenario, _, _ = combination
        title = f"{EVALUATED_MODELS[model_name]} ({SMOTE_SETTINGS[smote_setting]}), {scenario}"
        show = scenario == CLEAN_SCENARIO or show_noisy
        folder_path = os.path.join(matrices_folder_path, smote_setting, model_name)
        os.makedirs(folder_path, exist_ok=True)
        for population in ["all", "noisy"]:
            population_outcomes = select_population(combination_outcomes, population)
            if population_outcomes.empty:
                continue
            plot_confusion_matrix(
                get_confusion_matrix(population_outcomes, class_names),
                f"Confusion Matrix of {title}, {POPULATIONS[population]}",
                os.path.join(folder_path, f"{scenario}-{population}-{CONFUSION_MATRIX_PLOT_FILE_NAME}"),
                show=show,
            )

        noisy_outcomes = select_population(combination_outcomes, "noisy")
        if not noisy_outcomes.empty:
            plot_confusion_matrix(
                get_confusion_matrix(noisy_outcomes, class_names, CLEAN_PREDICTION_COLUMN, PREDICTION_COLUMN),
                f"Predictions of {title} on the Noisy Rows Against the Same Rows Without Noise",
                os.path.join(folder_path, f"{scenario}-{PREDICTION_TRANSITIONS_PLOT_FILE_NAME}"),
                show=show,
                xlabel="Prediction With Noise",
                ylabel="Prediction Without Noise",
            )
        print_progress(group_index, groups.ngroups, "Plotting confusion matrices", started=started)
    end_status(f"Saved the confusion matrices of {groups.ngroups} predictions to {matrices_folder_path}")

if SAVE_MATRIX_PLOTS:
    plot_every_matrix(outcomes, class_names)
```

Kode Program 5.114 Matriks konfusi dan matriks transisi prediksi

Kode Program 5.115 mengimplementasikan Langkah 7. Fungsi *load_prediction_summaries* membaca ringkasan setiap pengujian dari metadata file prediksi dan menghitung jumlah baris yang diprediksi per detik. Sel kedua menyimpan ringkasan tersebut pada file *prediction-summaries.csv*, menampilkan ringkasan pada data uji bersih, dan menampilkan waktu prediksi setiap model pada setiap data uji.

```python
def load_prediction_summaries() -> pd.DataFrame:
    summaries = []
    for smote_setting, model_name, scenario in list_predicted_combinations():
        summaries.append(read_prediction_summary(smote_setting, model_name, scenario))
    summaries = pd.DataFrame(summaries)
    summaries["rows_per_second"] = summaries[ROW_COUNT_COLUMN] / summaries["predict_seconds"]
    return summaries

prediction_summaries = load_prediction_summaries()
prediction_summaries.to_csv(os.path.join(PATH_FOLDER_EVALUATION, PREDICTION_SUMMARIES_FILE_NAME), index=False)
display(prediction_summaries[prediction_summaries[SCENARIO_COLUMN] == CLEAN_SCENARIO].round(4))
display(
    prediction_summaries.pivot_table(
        index=[SMOTE_SETTING_COLUMN, MODEL_COLUMN],
        columns=SCENARIO_COLUMN,
        values="predict_seconds",
    ).reindex(columns=list_scenario_names()).round(2)
)
```

Kode Program 5.115 Perbandingan biaya prediksi

Fungsi *find_key_findings* pada Kode Program 5.116 mengimplementasikan Langkah 8. Untuk setiap pengaturan SMOTE, fungsi ini menentukan model dengan F1-*score* makro bersih tertinggi, model dengan rata-rata retensi tertinggi dan terendah, skenario dan jenis gangguan dengan rata-rata retensi terendah pada populasi seluruh baris dan baris terganggu, model dengan laju perubahan prediksi, laju lolos serangan, laju alarm palsu, dan proporsi kesalahan yakin tertinggi pada baris terganggu, serta kelas serangan yang paling sering diprediksi *Benign* dan yang penurunan F1-*score*-nya terbesar pada baris terganggu. Fungsi ini juga menghitung selisih rata-rata retensi akibat SMOTE untuk setiap model, laju perubahan prediksi terbesar pada baris utuh, dan residu identitas campuran terbesar. Sel kedua menyimpan tabel temuan utama pada file *key-findings.csv* dan menampilkannya.

```python
def find_key_findings(
    metrics: pd.DataFrame,
    degradation: pd.DataFrame,
    class_metrics: pd.DataFrame,
    smote_effect: pd.DataFrame,
    mixture_check: pd.DataFrame,
    metric: str = ROBUSTNESS_METRIC,
) -> pd.DataFrame:
    findings = []
    is_noise_scenario = metrics[SCENARIO_COLUMN] != CLEAN_SCENARIO
    is_attack_class = class_metrics[CLASS_COLUMN] != BENIGN_CLASS
    for smote_setting, smote_label in SMOTE_SETTINGS.items():
        is_setting = metrics[SMOTE_SETTING_COLUMN] == smote_setting
        clean = metrics[is_setting & ~is_noise_scenario & (metrics[POPULATION_COLUMN] == "all")]
        if clean.empty:
            continue
        clean = clean.set_index(MODEL_COLUMN)[metric]
        best_clean_model = EVALUATED_MODELS[clean.idxmax()]
        findings.append((smote_label, f"Highest clean {METRIC_LABELS[metric]}", best_clean_model, clean.max()))

        for population in ["all", "noisy"]:
            is_shown = (
                (degradation[SMOTE_SETTING_COLUMN] == smote_setting)
                & (degradation[METRIC_COLUMN] == metric)
                & (degradation[POPULATION_COLUMN] == population)
                & (degradation[SCENARIO_COLUMN] != CLEAN_SCENARIO)
            )
            retention = degradation[is_shown]
            if retention.empty:
                continue
            by_model = retention.groupby(MODEL_COLUMN)[RETENTION_COLUMN].mean()
            by_scenario = retention.groupby(SCENARIO_COLUMN)[RETENTION_COLUMN].mean()
            by_noise_type = retention.groupby(NOISE_TYPE_COLUMN)[RETENTION_COLUMN].mean()
            rows_label = POPULATIONS[population].lower()
            findings.append((
                smote_label,
                f"Most robust model, {rows_label} (mean retention)",
                EVALUATED_MODELS[by_model.idxmax()],
                by_model.max(),
            ))
            findings.append((
                smote_label,
                f"Least robust model, {rows_label} (mean retention)",
                EVALUATED_MODELS[by_model.idxmin()],
                by_model.min(),
            ))
            findings.append((
                smote_label,
                f"Hardest scenario, {rows_label} (mean retention)",
                by_scenario.idxmin(),
                by_scenario.min(),
            ))
            findings.append((
                smote_label,
                f"Most damaging noise type, {rows_label} (mean retention)",
                by_noise_type.idxmin(),
                by_noise_type.min(),
            ))

        noisy_rows = metrics[is_setting & (metrics[POPULATION_COLUMN] == "noisy")]
        highest_questions = {
            "flip_rate": "Most unstable model on the noisy rows (mean flip rate)",
            "evasion_rate": "Model whose detected attacks turn Benign most often (mean evasion rate)",
            "false_alarm_rate": "Model with the most false alarms on the noisy rows (mean false alarm rate)",
            "confident_error_share": "Model with the most confident errors on the noisy rows",
        }
        for column, question in highest_questions.items():
            by_model = noisy_rows.groupby(MODEL_COLUMN)[column].mean().dropna()
            if not by_model.empty:
                findings.append((smote_label, question, EVALUATED_MODELS[by_model.idxmax()], by_model.max()))

        is_noisy_classes = (
            (class_metrics[SMOTE_SETTING_COLUMN] == smote_setting)
            & (class_metrics[POPULATION_COLUMN] == "noisy")
            & is_attack_class
        )
        noisy_classes = class_metrics[is_noisy_classes]
        if not noisy_classes.empty:
            benign_rate = noisy_classes.groupby(CLASS_COLUMN)["benign_rate"].mean().dropna()
            f1_drop = noisy_classes.groupby(CLASS_COLUMN)["f1_drop"].mean().dropna()
            findings.append((
                smote_label,
                "Attack most often predicted as Benign on the noisy rows",
                benign_rate.idxmax(),
                benign_rate.max(),
            ))
            findings.append((
                smote_label,
                "Attack with the largest F1-score drop on the noisy rows",
                f1_drop.idxmax(),
                f1_drop.max(),
            ))

    if not smote_effect.empty:
        is_shown = (
            (smote_effect[METRIC_COLUMN] == metric)
            & (smote_effect[POPULATION_COLUMN] == "all")
            & (smote_effect[SCENARIO_COLUMN] != CLEAN_SCENARIO)
        )
        retention_difference = smote_effect[is_shown].groupby(MODEL_COLUMN)["retention_difference"].mean()
        for model_name, difference in retention_difference.items():
            question = f"Mean retention with SMOTE minus without SMOTE, {EVALUATED_MODELS[model_name]}"
            findings.append(("Both", question, "", difference))

    is_intact = (metrics[POPULATION_COLUMN] == "intact") & is_noise_scenario
    findings.append((
        "Both",
        "Largest flip rate on the intact rows (0 means the clean baseline is valid)",
        "",
        metrics.loc[is_intact, "flip_rate"].max(),
    ))
    findings.append((
        "Both",
        "Largest accuracy mixture residual (0 means the populations add up)",
        "",
        mixture_check["residual"].abs().max(),
    ))
    return pd.DataFrame(findings, columns=["setting", "finding", "answer", "value"])

key_findings = find_key_findings(metrics, degradation, class_metrics, smote_effect, mixture_check)
key_findings.to_csv(os.path.join(PATH_FOLDER_EVALUATION, KEY_FINDINGS_FILE_NAME), index=False)
with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
    display(key_findings.round(4))
```

Kode Program 5.116 Penyusunan temuan utama
