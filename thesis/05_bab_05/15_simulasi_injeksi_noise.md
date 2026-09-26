## **5.15 Implementasi Simulasi Injeksi *Feature Noise* pada Data Uji**

Subbab ini menyajikan implementasi dari rancangan pada Subbab 4.15, yaitu pembentukan dua belas skenario gangguan pada data uji. Kode program disusun mengikuti Langkah 1 hingga 8 pada algoritma di Subbab 4.15, dilanjutkan dengan kode untuk memverifikasi hasil pembentukan skenario. Tahapan praproses pada Langkah 8 menggunakan kembali fungsi pada Subbab 5.4, 5.6, dan 5.7.

Kode Program 5.114 mengimplementasikan Langkah 1. Konstanta pada bagian awal menetapkan folder keluaran setiap tahap (*data-noise*, *data-noise-imputed*, *data-noise-scaled*, dan *data-noise-ipca*), empat jenis gangguan (*NOISE_TYPES*), tiga intensitas 0,10, 0,20, dan 0,30 (*NOISE_INTENSITIES*), himpunan yang diganggu (*NOISE_SPLIT*, yaitu *test*), *seed* dasar 42 (*NOISE_SEED*), dan cakupan gangguan per baris (*NOISE_SCOPE*). Kamus *NOISE_MAGNITUDES* menetapkan besar gangguan setiap jenis, yaitu 1,0 untuk *gaussian* dan *uniform*, 0,5 untuk *multiplicative*, dan tidak ada untuk *missing*. Konstanta *NOISE_UNIFORM_MATCH_VARIANCE* mengaktifkan penyamaan varians *uniform noise* dengan *gaussian noise*. Fungsi *get_noise_scenario_name* membentuk nama skenario dari jenis dan intensitas, *list_noise_scenarios* menyusun kedua belas skenario dari kombinasi jenis dan intensitas, dan *get_scenario_folder* membentuk lokasi folder suatu skenario.

```python
PATH_FOLDER_NOISE = "data-noise"
PATH_FOLDER_NOISE_IMPUTED = "data-noise-imputed"
PATH_FOLDER_NOISE_SCALED = "data-noise-scaled"
PATH_FOLDER_NOISE_IPCA = "data-noise-ipca"

NOISE_TYPES = ("gaussian", "uniform", "multiplicative", "missing")

NOISE_INTENSITIES = (0.10, 0.20, 0.30)
NOISE_SPLIT = "test"
NOISE_SEED = 42

NOISE_SCOPE = "row"

NOISE_MAGNITUDES = {
    "gaussian": 1.0,
    "uniform": 1.0,
    "multiplicative": 0.5,
    "missing": None,        
}

NOISE_UNIFORM_MATCH_VARIANCE = True

def get_noise_scenario_name(noise_type: str, intensity: float) -> str:
    return f"{noise_type}-{round(intensity * 100):02d}"


def list_noise_scenarios(
    noise_types: tuple[str, ...] = NOISE_TYPES,
    intensities: tuple[float, ...] = NOISE_INTENSITIES,
) -> list[tuple[str, float, str]]:
    return [
        (noise_type, intensity, get_noise_scenario_name(noise_type, intensity))
        for noise_type in noise_types
        for intensity in intensities
    ]


def get_scenario_folder(folder_name: str, scenario: str) -> str:
    return os.path.join(folder_name, scenario)
```

Kode Program 5.114 Pengaturan skenario gangguan dan penamaan skenario

Kode Program 5.115 mengimplementasikan Langkah 2 dan 3. Fungsi *get_feature_deviations* mengambil simpangan baku setiap fitur (*scale*) dari berkas statistik penskalaan sebagai acuan besar gangguan. Fungsi *make_file_rng* membentuk *seed* dari hasil *hash* BLAKE2b sepanjang 8 *byte* atas teks yang memuat *seed* dasar, nama skenario, dan nama berkas, kemudian membentuk generator *numpy.random.default_rng*.

```python
def get_feature_deviations(statistics_file: str = PATH_SCALER) -> pd.Series:
    return pd.Series(load_scaler_statistics(statistics_file)["scale"])

def make_file_rng(scenario: str, file_name: str, seed: int = NOISE_SEED) -> np.random.Generator:
    digest = hashlib.blake2b(f"{seed}/{scenario}/{file_name}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))
```

Kode Program 5.115 Simpangan baku fitur dan generator bilangan acak

Kode Program 5.116 mengimplementasikan Langkah 4. Fungsi *add_gaussian_noise* menambahkan bilangan acak berdistribusi normal dengan simpangan baku *magnitude* dikalikan simpangan baku fitur (Persamaan 4.5). Fungsi *add_uniform_noise* menambahkan bilangan acak berdistribusi seragam pada rentang simetris dengan setengah lebar *magnitude* dikalikan simpangan baku fitur dan, apabila penyamaan varians aktif, dikalikan √3 (Persamaan 4.6). Fungsi *add_multiplicative_noise* mengalikan nilai dengan bilangan acak berdistribusi normal dengan rata-rata 1 dan simpangan baku *magnitude* (Persamaan 4.7). Fungsi *blank_values* menghasilkan larik *NaN* untuk gangguan data tidak lengkap.

```python
def add_gaussian_noise(values, magnitude, deviations, rng):
    return values + rng.standard_normal(values.shape) * (magnitude * deviations)


def add_uniform_noise(values, magnitude, deviations, rng):
    half_width = magnitude * deviations * (math.sqrt(3.0) if NOISE_UNIFORM_MATCH_VARIANCE else 1.0)
    return values + rng.uniform(-1.0, 1.0, values.shape) * half_width


def add_multiplicative_noise(values, magnitude, deviations, rng):
    del deviations
    return values * rng.normal(1.0, magnitude, values.shape)


def blank_values(values, magnitude, deviations, rng):
    del magnitude, deviations, rng
    return np.full(values.shape, np.nan)
```

Kode Program 5.116 Pembangkitan nilai terganggu untuk setiap jenis gangguan

Fungsi *draw_coverage_mask* pada Kode Program 5.117 mengimplementasikan Langkah 5. Pada cakupan *row*, sebanyak pembulatan hasil kali intensitas dengan jumlah baris dipilih secara acak tanpa pengembalian, dan penanda baris terpilih diperluas ke seluruh kolom. Pada cakupan *cell*, sel individual dipilih dengan cara yang sama, dan cakupan ini tidak digunakan dalam penelitian ini.

```python
def draw_coverage_mask(
    shape: tuple[int, int], intensity: float, rng: np.random.Generator, scope: str = NOISE_SCOPE
) -> np.ndarray:
    n_rows, n_columns = shape

    if scope == "row":
        selected = np.zeros(n_rows, dtype=bool)
        selected[rng.choice(n_rows, round(intensity * n_rows), replace=False)] = True
        return np.repeat(selected[:, None], n_columns, axis=1)

    if scope == "cell":
        n_cells = n_rows * n_columns
        selected = np.zeros(n_cells, dtype=bool)
        selected[rng.choice(n_cells, round(intensity * n_cells), replace=False)] = True
        return selected.reshape(shape)

    raise ValueError(f"unknown NOISE_SCOPE {scope!r}; expected 'row' or 'cell'")
```

Kode Program 5.117 Pemilihan baris yang terdampak

Kamus *NOISE_INJECTORS* pada Kode Program 5.118 memetakan jenis gangguan ke fungsi pembangkitnya. Fungsi *add_noise_to_features* memeriksa bahwa jenis gangguan dikenal dan bahwa kolom tabel sama, baik nama maupun urutannya, dengan kolom yang dipelajari oleh *scaler*. Selanjutnya fungsi ini mengonversi fitur menjadi *float64*, membangkitkan nilai terganggu untuk seluruh sel, memilih baris yang terdampak, dan mengganti nilai hanya pada baris terpilih dengan *numpy.where* (Langkah 6).

```python
NOISE_INJECTORS = {
    "gaussian": add_gaussian_noise,
    "uniform": add_uniform_noise,
    "multiplicative": add_multiplicative_noise,
    "missing": blank_values,
}


def add_noise_to_features(
    df: pd.DataFrame,
    noise_type: str,
    intensity: float,
    deviations: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if noise_type not in NOISE_INJECTORS:
        raise ValueError(f"unknown noise type {noise_type!r}; expected one of {sorted(NOISE_INJECTORS)}")

    if list(df.columns) != list(deviations.index):
        raise ValueError("this file's columns are not, in order, the columns the scaler was fitted on")

    values = df.to_numpy(dtype="float64")
    damaged = NOISE_INJECTORS[noise_type](
        values, NOISE_MAGNITUDES[noise_type], deviations.to_numpy(), rng
    )
    selected = draw_coverage_mask(values.shape, intensity, rng)

    return pd.DataFrame(np.where(selected, damaged, values), columns=df.columns)
```

Kode Program 5.118 Penerapan gangguan pada satu tabel fitur

Fungsi *add_noise_to_file* pada Kode Program 5.119 membaca satu berkas data uji terimputasi, memanggil *add_noise_to_features* dengan generator acak khusus skenario dan berkas tersebut, dan menulis hasilnya ke folder tujuan dengan nama berkas yang sama (Langkah 7).

```python
def add_noise_to_file(
    source_folder_name: str,
    target_folder_name: str,
    file_name: str,
    scenario: str,
    noise_type: str,
    intensity: float,
    deviations: pd.Series,
) -> str:
    df = read_parquet_file(source_folder_name, file_name)
    noisy_features = add_noise_to_features(
        df, noise_type, intensity, deviations, make_file_rng(scenario, file_name)
    )
    return write_dataframe_to_parquet(noisy_features, target_folder_name, file_name)
```

Kode Program 5.119 Pemberian gangguan pada satu berkas

Fungsi *add_noise_to_split* pada Kode Program 5.120 memuat simpangan baku fitur, kemudian menjalankan *add_noise_to_file* pada seluruh berkas himpunan uji melalui *process_all_splits* (Kode Program 5.24), sehingga berkas diproses secara paralel.

```python
def add_noise_to_split(
    source_folder_name: str,
    target_folder_name: str,
    scenario: str,
    noise_type: str,
    intensity: float,
    statistics_file: str = PATH_SCALER,
    split_name: str = NOISE_SPLIT,
) -> None:
    deviations = get_feature_deviations(statistics_file)
    process_all_splits(
        source_folder_name,
        target_folder_name,
        (split_name,),
        add_noise_to_file,
        scenario,
        noise_type,
        intensity,
        deviations,
    )
```

Kode Program 5.120 Pemberian gangguan pada seluruh berkas satu skenario

Fungsi *count_scenario_files* pada Kode Program 5.121 menghitung jumlah berkas keluaran suatu skenario. Fungsi *build_noise_scenario* mengimplementasikan Langkah 8. Fungsi ini melewati skenario yang jumlah berkas keluarannya telah lengkap, kecuali apabila parameter *rebuild* bernilai *True*, kemudian memberi gangguan pada data uji terimputasi dan melewatkan hasilnya melalui *impute_all_splits*, *scale_all_splits*, dan *project_all_splits*. Ketiga fungsi tersebut dibatasi pada himpunan uji, dan proyeksi menggunakan *IPCA_COMPONENTS* komponen, sehingga tahapan praproses yang digunakan sama dengan Subbab 5.4, 5.6, dan 5.7.

```python
def count_scenario_files(
    folder_name: str, scenario: str, split_name: str = NOISE_SPLIT
) -> int:
    folder = get_split_folder(get_scenario_folder(folder_name, scenario), split_name)
    if not os.path.isdir(folder):
        return 0
    return len(get_all_file_names_in_folder(folder, "parquet"))


def build_noise_scenario(
    noise_type: str,
    intensity: float,
    scenario: str,
    source_folder_name: str = PATH_FOLDER_IMPUTED,
    n_components: int = None,
    split_name: str = NOISE_SPLIT,
    rebuild: bool = False,
) -> None:
    n_components = IPCA_COMPONENTS if n_components is None else n_components
    expected = len(get_all_file_names_in_folder(
        get_split_folder(source_folder_name, split_name), "parquet"
    ))

    if not rebuild and count_scenario_files(PATH_FOLDER_NOISE_IPCA, scenario, split_name) == expected:
        print(f"[{scenario}] already built ({expected} files), skipped.")
        return

    print(f"[{scenario}] {noise_type} at {intensity:.0%}, {expected} files")

    add_noise_to_split(
        source_folder_name,
        get_scenario_folder(PATH_FOLDER_NOISE, scenario),
        scenario,
        noise_type,
        intensity,
        split_name=split_name,
    )
    impute_all_splits(
        get_scenario_folder(PATH_FOLDER_NOISE, scenario),
        get_scenario_folder(PATH_FOLDER_NOISE_IMPUTED, scenario),
        split_names=(split_name,),
    )
    scale_all_splits(
        get_scenario_folder(PATH_FOLDER_NOISE_IMPUTED, scenario),
        get_scenario_folder(PATH_FOLDER_NOISE_SCALED, scenario),
        split_names=(split_name,),
    )
    project_all_splits(
        get_scenario_folder(PATH_FOLDER_NOISE_SCALED, scenario),
        get_scenario_folder(PATH_FOLDER_NOISE_IPCA, scenario),
        n_components,
        split_names=(split_name,),
    )
```

Kode Program 5.121 Pembentukan satu skenario gangguan lengkap

Fungsi *build_all_noise_scenarios* pada Kode Program 5.122 membentuk setiap skenario pada daftar secara berurutan dan mencetak jumlah skenario yang tersedia.

```python
def build_all_noise_scenarios(
    scenarios: list[tuple[str, float, str]] | None = None,
    source_folder_name: str = PATH_FOLDER_IMPUTED,
    rebuild: bool = False,
) -> None:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios

    for noise_type, intensity, scenario in scenarios:
        build_noise_scenario(
            noise_type, intensity, scenario, source_folder_name, rebuild=rebuild
        )

    print(f"\n{len(scenarios)} scenarios under {PATH_FOLDER_NOISE_IPCA}/.")
```

Kode Program 5.122 Pembentukan seluruh skenario gangguan

Kode Program 5.123 menjalankan pembentukan kedua belas skenario dari folder *data-imputed* ke folder *data-noise-ipca*.

```python
build_all_noise_scenarios()
```

Kode Program 5.123 Menjalankan pembentukan seluruh skenario gangguan

Kode Program 5.124 mengimplementasikan verifikasi hasil pembentukan skenario. Fungsi *measure_noise_in_frames* membandingkan tabel bersih dan tabel ternoise, kemudian menghitung proporsi baris yang berubah, proporsi sel yang berubah, dan besar gangguan yang teramati. Besar gangguan yang teramati adalah simpangan baku selisih nilai dalam satuan simpangan baku fitur pada *gaussian* dan *uniform*, atau simpangan baku perubahan relatif pada *multiplicative*, dan tidak diukur pada *missing*. Fungsi *summarize_noise_injection* menerapkannya pada berkas pertama himpunan uji untuk setiap skenario, dan sel terakhir menjalankannya.

```python
def measure_noise_in_frames(
    clean: pd.DataFrame, noisy: pd.DataFrame, noise_type: str, deviations: pd.Series
) -> dict:
    clean_values = clean.to_numpy(dtype="float64")
    noisy_values = noisy.to_numpy(dtype="float64")

    changed = ~(noisy_values == clean_values)

    if noise_type == "missing":
        magnitude = float("nan")
    elif noise_type == "multiplicative":
        touched = changed & (clean_values != 0.0)
        magnitude = float(
            np.std((noisy_values[touched] - clean_values[touched]) / clean_values[touched])
        )
    else:
        scaled = (noisy_values - clean_values) / deviations.to_numpy()
        magnitude = float(np.std(scaled[changed]))

    return {
        "records changed": float(changed.any(axis=1).mean()),
        "cells changed": float(changed.mean()),
        "magnitude": magnitude,
    }

def summarize_noise_injection(
    scenarios: list[tuple[str, float, str]] | None = None,
    source_folder_name: str = PATH_FOLDER_IMPUTED,
    noise_folder_name: str = PATH_FOLDER_NOISE,
    split_name: str = NOISE_SPLIT,
    statistics_file: str = PATH_SCALER,
) -> pd.DataFrame:
    scenarios = list_noise_scenarios() if scenarios is None else scenarios
    deviations = get_feature_deviations(statistics_file)

    clean_folder = get_split_folder(source_folder_name, split_name)
    file_name = get_all_file_names_in_folder(clean_folder, "parquet")[0]
    clean = read_parquet_file(clean_folder, file_name)

    rows = []
    for noise_type, intensity, scenario in scenarios:
        noisy_folder = get_split_folder(get_scenario_folder(noise_folder_name, scenario), split_name)
        noisy = read_parquet_file(noisy_folder, file_name)
        rows.append(
            {
                "scenario": scenario,
                "noise": noise_type,
                "intensity": intensity,
                "magnitude set": NOISE_MAGNITUDES[noise_type],
                **measure_noise_in_frames(clean, noisy, noise_type, deviations),
            }
        )

    return pd.DataFrame(rows)

summarize_noise_injection()
```

Kode Program 5.124 Verifikasi hasil pembentukan skenario gangguan
