import pytest
from pyspark.sql import SparkSession

from pyspark_job import clean_data


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("PySpark-CI-Tests")
        .getOrCreate()
    )

    yield session
    session.stop()


@pytest.fixture
def sample_df(spark):
    data = [
        ("Ahmed", 100.0),
        ("Mona", 50.0),
        ("Ali", 0.0),
        ("Sara", -10.0),
        (None, 80.0),
    ]

    return spark.createDataFrame(
        data,
        ["name", "amount"]
    )


def test_valid_records_are_kept(sample_df):
    result = clean_data(sample_df)

    names = {row["name"] for row in result.collect()}

    assert names == {"Ahmed", "Mona"}


def test_non_positive_amounts_are_removed(sample_df):
    result = clean_data(sample_df)

    amounts = [row["amount"] for row in result.collect()]

    assert all(amount > 0 for amount in amounts)


def test_null_names_are_removed(sample_df):
    result = clean_data(sample_df)

    assert all(
        row["name"] is not None
        for row in result.collect()
    )


def test_amount_with_tax_is_calculated(sample_df):
    result = clean_data(sample_df)

    amounts = {
        row["name"]: row["amount_with_tax"]
        for row in result.collect()
    }

    assert amounts["Ahmed"] == pytest.approx(120.0)
    assert amounts["Mona"] == pytest.approx(60.0)
