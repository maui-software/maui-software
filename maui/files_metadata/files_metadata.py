import re
import yaml
import datetime
from importlib import resources


def verify_yaml_format(data):
    """
    Verify if the provided YAML data follows a specific format.

    Parameters
    ----------
    data : dict
        A dictionary representing YAML data.

    Returns
    -------
    bool
        True if the YAML data follows the expected format, False otherwise.
    """

    if 'formats' not in data or not isinstance(data['formats'], list):
        # If 'formats' key is missing or its value is not a list, return False
        return False

    for format_data in data['formats']:
        # Iterate through each format data in the 'formats' list
        if ('format_name' not in format_data or 'file_name_format' not in format_data
            or 'file_extension' not in format_data or  'metadata_tag_info' not in format_data):
        
            # If any of the required keys are missing in format data, return False
            return False

        metadata_tag_info = format_data['metadata_tag_info']

        for _, tag_info in metadata_tag_info.items():
            # Iterate through each metadata tag info in the 'metadata_tag_info' dictionary
            if 'description' not in tag_info or 'type' not in tag_info or 'format' not in tag_info:
                # If any of the required keys are missing in tag info, return False
                return False

    # If all checks pass, return True indicating the YAML data follows the expected format
    return True

# ---------------------------------------------------

def get_format_config(format_name, format_file_path):
    """
    Retrieve configuration for a specific format from a YAML file.

    Parameters
    ----------
    format_name : str
        Name of the format to retrieve configuration for.
    format_file_path : str
        Path to the YAML file containing format configurations.

    Returns
    -------
    dict
        A dictionary containing configuration information for the specified format.

    Raises
    ------
    ValueError
        If the provided YAML file is not properly formatted or if the specified format is not found.
    """

    # Load data from .yaml
    if format_file_path is None:
        with resources.open_text('maui.files_metadata', 'files_formats.yaml') as file:
            data = yaml.safe_load(file)
    else:
        with open(format_file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Verify if YAML data follows the expected format
    if not verify_yaml_format(data):
        raise ValueError("The provided YAML is not properly formatted")

    # Search for the format with provided name
    for format_data in data['formats']:
        if format_data['format_name'] == format_name:
            selected_format = format_data
            break
    else:
        raise ValueError(f"{format_name} not found in the YAML data")

    return selected_format

# ---------------------------------------------------

def extract_metadata(string, format_name, date_time_func=None, format_file_path=None):
    """
    Extract metadata from a string based on a specified format.

    Parameters
    ----------
    string : str
        The string from which metadata will be extracted.
    format_name : str
        Name of the format to use for metadata extraction.
    date_time_func : function, optional
        A function to handle date and time processing for extracted metadata.
        Default is None.
    format_file_path : str, optional
        Path to the YAML file containing format configurations.
        Default is 'files_formats.yaml'.

    Returns
    -------
    dict or None
        A dictionary containing extracted metadata if successful, None otherwise.

    Raises
    ------
    ValueError
        If the specified format is not found in the format file.
    """
    # Retrieve format configuration from YAML file
    file_format_config = get_format_config(format_name, format_file_path)

    # Extract pattern and metadata dictionary from format configuration
    pattern = file_format_config['file_name_format']
    metadata_dict = file_format_config['metadata_tag_info']

    # Fill pattern with metadata format placeholders
    pattern_filled = pattern
    for key in metadata_dict.keys():
        pattern_filled = pattern_filled.replace(key, metadata_dict[key]['format'])

    # Compile regex pattern and match against input string
    regex = re.compile(pattern_filled)
    result = regex.match(string)

    # If match is found, extract metadata values and return as a dictionary
    if result:
        values = result.groups()
        values = dict(zip(metadata_dict.keys(), values))

        # If the format is "LEEC_FILE_FORMAT", handle specific date and time format
        if format_name == "LEEC_FILE_FORMAT":
            values["timestamp_init"] = datetime.datetime.strptime(values['date'] + ' ' + values['time'], "%Y%m%d %H%M%S")
        # If a date_time_func is provided, apply it to the metadata values
        elif date_time_func is not None:
            values = date_time_func(values)

        return values
    else:
        return None