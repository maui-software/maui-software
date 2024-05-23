"""
Module containing tests for files_metadata module.

This module provides test cases for the files_metadata module in the Maui software.
"""

import datetime

import pytest
from maui import files_metadata


# Fixtures
@pytest.fixture(name="sample_yaml_path")
def sample_yaml(tmp_path):
    """
    Fixture to create a sample YAML file for testing.

    Parameters
    ----------
    tmp_path : str
        Temporary directory path.

    Returns
    -------
    str
        Path to the created sample YAML file.
    """
    # Create a sample YAML file for testing
    yaml_content = """
    formats:
      - format_name: "LEEC_FILE_FORMAT_CUSTOM"
        file_name_format: landscape__channel__date_time_environment
        file_extension: wav
        metadata_tag_info:
          landscape: 
            description: Determine which site the recording equipment is placed in a given date and time. The mapping of each landscape and its position is done in another file.
            type: String
            format: '([A-Za-zÀ-ü0-9]+)'
          channel:
            description: Recording channel used in the equipment.
            type: Integer
            format: '(\\d+)'
          date:
            description: Date in which the record started.
            type: Date
            format: '(\\d{8})'
            date_format: 'YYYYMMDD'
          time:
            description: Time of the day in which the record started.
            type: Time
            format: '(\\d{6})'
            time_format: 'hhmmss'
          environment:
            description: Type of environment of the recording
            type: String
            format: '([A-Za-zÀ-ü]+)'

    """
    yaml_file = tmp_path / "sample.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_file


@pytest.fixture(name="sample_yaml_invalid_path")
def sample_yaml_invalid(tmp_path):
    """
    Fixture to create a sample invalid YAML file for testing.

    Parameters
    ----------
    tmp_path : str
        Temporary directory path.

    Returns
    -------
    str
        Path to the created sample invalid YAML file.
    """
    # Create a sample YAML file for testing
    yaml_content = """
    formats:
      - format_name: LEEC_FILE_FORMAT
        file_name_format: csv_[timestamp]
        metadata_tag_info:
          id:
            description: Unique identifier
            type: integer
          name:
            description: Name of the item
            type: string
            format: default
    """
    yaml_file = tmp_path / "sample_invalid.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_file


@pytest.fixture(name="sample_string_leec")
def sample_string():
    """
    Fixture to provide a sample string for testing.

    Returns
    -------
    str
        Sample string.
    """
    return "LEEC40__0__20170111_204600_br.wav"


# ---------------------------------------------------------


@pytest.mark.parametrize(
    "data, expected_result",
    [
        # Valid YAML format
        (
            {
                "formats": [
                    {
                        "format_name": "Example Format",
                        "file_name_format": "example_{id}",
                        "file_extension": ".txt",
                        "metadata_tag_info": {
                            "tag1": {
                                "description": "Description 1",
                                "type": "Type 1",
                                "format": "Format 1",
                            },
                            "tag2": {
                                "description": "Description 2",
                                "type": "Type 2",
                                "format": "Format 2",
                            },
                        },
                    }
                ]
            },
            True,
        ),
        # Missing 'formats' key
        ({}, False),
        # Missing required keys in format data
        ({"formats": [{"format_name": "Example Format"}]}, False),
        # Missing required keys in tag info
        (
            {
                "formats": [
                    {
                        "format_name": "Example Format",
                        "file_name_format": "example_{id}",
                        "file_extension": ".txt",
                        "metadata_tag_info": {
                            "tag1": {"description": "Description 1", "type": "Type 1"}
                        },
                    }
                ]
            },
            False,
        ),
    ],
)
def test_verify_yaml_format(data, expected_result):
    """
    Test function for verify_yaml_format.

    Parameters
    ----------
    data : dict
        Input data for testing.
    expected_result : bool
        Expected result of the function.

    Returns
    -------
    None
    """
    assert files_metadata.verify_yaml_format(data) == expected_result


# ---------------------------------------------------------


def test_get_format_config_with_valid_format(sample_yaml_path):
    """
    Test function for get_format_config with a valid format.

    Parameters
    ----------
    sample_yaml_path : str
        Path to the sample YAML file.

    Returns
    -------
    None
    """
    config = files_metadata.get_format_config("LEEC_FILE_FORMAT_CUSTOM", sample_yaml_path)
    expected_config = {
        "format_name": "LEEC_FILE_FORMAT_CUSTOM",
        "file_name_format": "landscape__channel__date_time_environment",
        "file_extension": "wav",
        "metadata_tag_info": {
            "landscape": {
                "description": "Determine which site the recording equipment "\
                "is placed in a given date and time. The mapping of each "\
                "landscape and its position is done in another file.",
                "type": "String",
                "format": "([A-Za-zÀ-ü0-9]+)",
            },
            "channel": {
                "description": "Recording channel used in the equipment.",
                "type": "Integer",
                "format": "(\\d+)",
            },
            "date": {
                "description": "Date in which the record started.",
                "type": "Date",
                "format": "(\\d{8})",
                "date_format": "YYYYMMDD",
            },
            "time": {
                "description": "Time of the day in which the record started.",
                "type": "Time",
                "format": "(\\d{6})",
                "time_format": "hhmmss",
            },
            "environment": {
                "description": "Type of environment of the recording",
                "type": "String",
                "format": "([A-Za-zÀ-ü]+)",
            },
        },
    }
    assert config == expected_config


def test_get_format_config_with_invalid_format(sample_yaml_path):
    """
    Test function for get_format_config with an invalid format.

    Parameters
    ----------
    sample_yaml_path : str
        Path to the sample YAML file.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        files_metadata.get_format_config("TEST", sample_yaml_path)


def test_get_format_config_with_invalid_yaml(sample_yaml_invalid_path):
    """
    Test function for get_format_config with an invalid YAML.

    Parameters
    ----------
    sample_yaml_invalid_path : str
        Path to the sample invalid YAML file.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        files_metadata.get_format_config("LEEC_FILE_FORMAT_CUSTOM", sample_yaml_invalid_path)


def test_get_format_config_with_no_file_path_valid_format():
    """
    Test function for get_format_config with no file path and a valid format.

    Returns
    -------
    None
    """
    config = files_metadata.get_format_config("LEEC_FILE_FORMAT", None)

    expected_config = {
        "format_name": "LEEC_FILE_FORMAT",
        "file_name_format": "landscape__channel__date_time_environment",
        "file_extension": "wav",
        "metadata_tag_info": {
            "landscape": {
                "description": "Determine which site the recording equipment "\
                "is placed in a given date and time. The mapping of each "\
                "landscape and its position is done in another file.",
                "type": "String",
                "format": "([A-Za-zÀ-ü0-9]+)",
            },
            "channel": {
                "description": "Recording channel used in the equipment.",
                "type": "Integer",
                "format": "(\\d+)",
            },
            "date": {
                "description": "Date in which the record started.",
                "type": "Date",
                "format": "(\\d{8})",
                "date_format": "YYYYMMDD",
            },
            "time": {
                "description": "Time of the day in which the record started.",
                "type": "Time",
                "format": "(\\d{6})",
                "time_format": "hhmmss",
            },
            "environment": {
                "description": "Type of environment of the recording",
                "type": "String",
                "format": "([A-Za-zÀ-ü]+)",
            },
        },
    }
    assert config == expected_config


def test_get_format_config_with_no_file_path_invalid_format():
    """
    Test function for get_format_config with no file path and an invalid format.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        files_metadata.get_format_config("TEST", None)


# ---------------------------------------------------------


def test_extract_metadata_with_valid_format_standard(sample_string_leec):
    """
    Test function for extract_metadata with a valid standard format.

    Parameters
    ----------
    sample_string_leec : str
        Sample string.

    Returns
    -------
    None
    """
    metadata = files_metadata.extract_metadata(
        sample_string_leec, "LEEC_FILE_FORMAT", format_file_path=None
    )
    assert metadata == {
        "landscape": "LEEC40",
        "channel": "0",
        "date": "20170111",
        "time": "204600",
        "environment": "br",
        "timestamp_init": datetime.datetime(2017, 1, 11, 20, 46),
    }


def test_extract_metadata_with_valid_format_custom(sample_string_leec, sample_yaml_path):
    """
    Test function for extract_metadata with a valid custom format.

    Parameters
    ----------
    sample_string_leec : str
        Sample string.
    sample_yaml_path : str
        Path to the sample YAML file.

    Returns
    -------
    None
    """
    metadata = files_metadata.extract_metadata(
        sample_string_leec, "LEEC_FILE_FORMAT_CUSTOM", format_file_path=sample_yaml_path
    )
    assert metadata == {
        "landscape": "LEEC40",
        "channel": "0",
        "date": "20170111",
        "time": "204600",
        "environment": "br",
    }


def test_extract_metadata_with_invalid_format():
    """
    Test function for extract_metadata with an invalid format.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        files_metadata.extract_metadata("invalid_string", "UNKNOWN_FORMAT")


def test_extract_metadata_with_date_time_func(sample_string_leec, sample_yaml_path):
    """
    Test function for extract_metadata with a date time function.

    Parameters
    ----------
    sample_string_leec : str
        Sample string.
    sample_yaml_path : str
        Path to the sample YAML file.

    Returns
    -------
    None
    """
    # Define a custom date time function for testing
    def custom_date_time_func(values):
        values["timestamp_init"] = datetime.datetime.strptime(
            values["date"] + " " + values["time"], "%Y%m%d %H%M%S"
        )
        return values

    metadata = files_metadata.extract_metadata(
        sample_string_leec,
        "LEEC_FILE_FORMAT_CUSTOM",
        format_file_path=sample_yaml_path,
        date_time_func=custom_date_time_func,
    )
    assert metadata == {
        "landscape": "LEEC40",
        "channel": "0",
        "date": "20170111",
        "time": "204600",
        "environment": "br",
        "timestamp_init": datetime.datetime(2017, 1, 11, 20, 46),
    }


def test_extract_metadata_with_no_match(sample_yaml_path):
    """
    Test function for extract_metadata with no match.

    Parameters
    ----------
    sample_yaml_path : str
        Path to the sample YAML file.

    Returns
    -------
    None
    """
    metadata = files_metadata.extract_metadata(
        "invalid_string", "LEEC_FILE_FORMAT_CUSTOM", format_file_path=sample_yaml_path
    )
    assert metadata is None
