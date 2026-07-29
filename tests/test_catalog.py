"""Unit tests for pyfracval.catalog -- loading and filtering
cluster_index.csv-style master indexes."""

from pathlib import Path

from pyfracval.catalog import filter_catalog, load_catalog

_HEADER = "config,sigma,Df,N,kf,attempt,seed,success,filepath\n"


def _write_index(path: Path, rows: list[str]) -> Path:
    path.write_text(_HEADER + "\n".join(rows) + "\n")
    return path


def test_load_catalog_parses_all_fields(tmp_path):
    index = _write_index(
        tmp_path / "cluster_index.csv",
        [
            "vanilla,1.0,2.0,32,0.8,0,12345,True,"
            "/orig/machine/vanilla/sigma_1p00__Df_2p00__N_32/agg0.dat"
        ],
    )
    entries = load_catalog(index)
    assert len(entries) == 1
    e = entries[0]
    assert e.config == "vanilla"
    assert e.sigma == 1.0
    assert e.Df == 2.0
    assert e.N == 32
    assert e.kf == 0.8
    assert e.attempt == 0
    assert e.seed == 12345
    assert e.success is True


def test_load_catalog_rebase_path_uses_last_three_components(tmp_path):
    """The stored filepath is from the original generation machine and
    isn't portable -- when data_root is given, only the last 3 path
    components (config/sigma_.._Df_.._N_../filename) are trusted, and
    re-joined under data_root, regardless of the absolute prefix stored."""
    index = _write_index(
        tmp_path / "cluster_index.csv",
        [
            "vanilla,1.0,2.0,32,0.8,0,12345,True,"
            "/some/other/machine/vanilla/sigma_1p00__Df_2p00__N_32/agg0.dat"
        ],
    )
    entries = load_catalog(index, data_root=tmp_path / "my_data")
    assert entries[0].filepath == (
        tmp_path / "my_data" / "vanilla" / "sigma_1p00__Df_2p00__N_32" / "agg0.dat"
    )


def test_load_catalog_without_data_root_uses_filepath_verbatim(tmp_path):
    index = _write_index(
        tmp_path / "cluster_index.csv",
        ["vanilla,1.0,2.0,32,0.8,0,12345,True,/orig/agg0.dat"],
    )
    entries = load_catalog(index)
    assert entries[0].filepath == Path("/orig/agg0.dat")


def test_load_catalog_parses_success_false(tmp_path):
    index = _write_index(
        tmp_path / "cluster_index.csv",
        ["vanilla,1.0,2.0,32,0.8,0,12345,False,/orig/agg0.dat"],
    )
    assert load_catalog(index)[0].success is False


def _sample_entries():
    index_rows = [
        ("vanilla", 1.0, 2.0, 32, 0.8, True),
        ("vanilla", 1.25, 2.0, 32, 0.8, True),
        ("vanilla", 1.5, 2.2, 64, 1.0, True),
        ("densify_retry", 1.0, 2.0, 128, 0.8, True),
        ("densify_retry", 1.0, 2.0, 8, 0.8, False),  # failed attempt
    ]
    return [
        f"{config},{sigma},{df},{n},{kf},0,1,{success},/x/agg.dat"
        for config, sigma, df, n, kf, success in index_rows
    ]


def test_filter_catalog_by_sigma_scalar(tmp_path):
    index = _write_index(tmp_path / "cluster_index.csv", _sample_entries())
    entries = load_catalog(index)
    filtered = filter_catalog(entries, sigma=1.0)
    assert {e.config for e in filtered} == {"vanilla", "densify_retry"}
    assert len(filtered) == 2  # the sigma=1.0/N=8 row is filtered by success_only


def test_filter_catalog_by_sigma_sequence(tmp_path):
    index = _write_index(tmp_path / "cluster_index.csv", _sample_entries())
    entries = load_catalog(index)
    filtered = filter_catalog(entries, sigma=[1.0, 1.25])
    assert len(filtered) == 3
    assert all(e.sigma in (1.0, 1.25) for e in filtered)


def test_filter_catalog_success_only_excludes_failures_by_default(tmp_path):
    index = _write_index(tmp_path / "cluster_index.csv", _sample_entries())
    entries = load_catalog(index)
    assert all(e.success for e in filter_catalog(entries))


def test_filter_catalog_success_only_false_includes_failures(tmp_path):
    index = _write_index(tmp_path / "cluster_index.csv", _sample_entries())
    entries = load_catalog(index)
    filtered = filter_catalog(entries, success_only=False, N=8)
    assert len(filtered) == 1
    assert filtered[0].success is False


def test_filter_catalog_combines_filters_as_and(tmp_path):
    index = _write_index(tmp_path / "cluster_index.csv", _sample_entries())
    entries = load_catalog(index)
    filtered = filter_catalog(entries, config="vanilla", sigma=1.5)
    assert len(filtered) == 1
    assert filtered[0].N == 64


def test_filter_catalog_no_filters_returns_all_successful(tmp_path):
    index = _write_index(tmp_path / "cluster_index.csv", _sample_entries())
    entries = load_catalog(index)
    assert len(filter_catalog(entries)) == 4  # 5 rows, 1 failed
