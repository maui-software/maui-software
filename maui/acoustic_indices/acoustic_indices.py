"""
    This module is designed to calculate various acoustic indices from audio files,
    providing insights into the characteristics of soundscapes. Utilizing the MAAD
    library for the extraction of acoustic features, it supports a wide range of
    indices, including but not limited to median amplitude envelope, temporal
    entropy, acoustic richness, and spectral entropy. These indices are essential
    for analyzing environmental sounds, bioacoustics data, and general acoustic
    properties of audio recordings.

    The core functionality is facilitated through a high-level function that
    accepts a pandas DataFrame containing paths to audio files, a list of indices
    to calculate, and options for parallel processing to expedite calculations
    across multiple audio files. Additional utility functions support parallel
    dataframe processing and interaction with filesystem for result storage.

    Features include:
    - Calculation of a comprehensive set of acoustic indices.
    - Parallel processing capabilities for efficiency.
    - Integration with the MAAD library for acoustic feature extraction.
    - Options to save calculated indices to various file formats.

    Functions:
    - calculate_acoustic_indices: Main function to calculate specified acoustic
      indices for a collection of audio files.
    - get_acoustic_indices: Calculates acoustic indices for individual audio files.
    - parallelize_dataframe: Splits DataFrame for parallel processing of audio
      files.
    - calculate_acoustic_indices_par_aux: Auxiliary function for parallel
      processing.

    Usage of this module is intended for researchers, ecologists, and sound
    designers interested in quantifying the acoustic properties of soundscapes
    or audio collections for analysis, monitoring, and other scientific studies.

    Dependencies:
    - numpy and pandas for data manipulation.
    - tqdm for progress tracking during calculations.
    - maad for acoustic feature extraction.
    - multiprocessing for parallel computation.

    Example usage and additional parameter descriptions are provided within each
    function's docstring, offering guidance on applying these tools to your audio
    analysis workflows.
"""

import gc
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
import maad

import maui.io


def get_acoustic_indices(df: pd.DataFrame, args: dict[str, any]) -> pd.DataFrame:

    indices_list = args["indices_list"]

    # create temp lit of indices to keep appending for each row
    indices_temp = {}

    for ind in indices_list:
        if ind == "median_amplitude_envelope":
            indices_temp["m"] = []
        if ind == "temporal_entropy":
            indices_temp["ht"] = []
        if ind == "temporal_activity":
            indices_temp["act_fract"] = []
            indices_temp["act_count"] = []
            indices_temp["act_mean"] = []
        if ind == "temporal_events":
            indices_temp["evnt_fract"] = []
            indices_temp["evn_mean"] = []
            indices_temp["evn_count"] = []
        if ind == "frequency_entropy":
            indices_temp["hf"] = []
            indices_temp["ht_per_bin"] = []
        if ind == "number_of_peaks":
            indices_temp["nb_peaks"] = []
        if ind == "spectral_entropy":
            indices_temp["eas"] = []
            indices_temp["ecu"] = []
            indices_temp["ecv"] = []
            indices_temp["eps"] = []
            indices_temp["eps_kurt"] = []
            indices_temp["eps_skew"] = []
        if ind == "spectral_activity":
            indices_temp["act_sp_fract_per_bin"] = []
            indices_temp["act_sp_count_per_bin"] = []
            indices_temp["act_sp_mean_per_bin"] = []
            indices_temp["Meanact_sp_fract_per_bin"] = []
            indices_temp["Meanact_sp_count_per_bin"] = []
        if ind == "spectral_events":
            indices_temp["mean_evn_sp_fract_per_bin"] = []
            indices_temp["evn_sp_fract_per_bin"] = []
            indices_temp["evn_sp_mean_per_bin"] = []
            indices_temp["evn_sp_count_per_bin"] = []
            indices_temp["evn_sp"] = []
        if ind == "spectral_cover":
            indices_temp["lfc"] = []
            indices_temp["mfc"] = []
            indices_temp["hfc"] = []
        if ind == "soundscape_index":
            indices_temp["ndsi"] = []
            indices_temp["ratio_ba"] = []
            indices_temp["antro_ph"] = []
            indices_temp["bio_ph"] = []
        if ind == "acoustic_diversity_index":
            indices_temp["adi"] = []
        if ind == "acoustic_eveness_index":
            indices_temp["aei"] = []
        if ind == "temporal_leq":
            indices_temp["leq"] = []
        if ind == "spectral_leq":
            indices_temp["leqf"] = []
            indices_temp["leqf_per_bin"] = []
        if ind == "tfsd":
            indices_temp["tfsd"] = []
        if ind == "more_entropy":
            indices_temp["ht_havrda"] = []
            indices_temp["ht_renyi"] = []
            indices_temp["ht_paired_shannon"] = []
            indices_temp["ht_gamma"] = []
            indices_temp["ht_gini_simpson"] = []
            indices_temp["hf_havrda"] = []
            indices_temp["hf_renyi"] = []
            indices_temp["hf_paired_shannon"] = []
            indices_temp["hf_gamma"] = []
            indices_temp["hf_gini_simpson"] = []
        if ind == "acoustic_gradient_index":
            indices_temp["agi_xx"] = []
            indices_temp["agi_per_bin"] = []
            indices_temp["agi_mean"] = []
            indices_temp["agi_sum"] = []
        if ind == "frequency_raoq":
            indices_temp["raoq"] = []
        if ind == "region_of_interest_index":
            indices_temp["roi_total"] = []
            indices_temp["roi_cover"] = []
        if ind == "acoustic_complexity_index":
            indices_temp["aci_sum"] = []
            indices_temp["aci_xx"] = []
            indices_temp["aci_per_bin"] = []

    # for each row of the dataframe, a list of acoustic indices
    # (given by indices_list parameter) is calculated
    iterator_df = range(df.shape[0]) if args["parallel"] else tqdm(range(df.shape[0]))

    for index in iterator_df:
        row = df.iloc[index]

        # load audio file
        s, fs = maad.sound.load(row["file_path"])
        if len(s) == 0:
            for key, value in indices_temp.items():
                value.append(None)

        else:

            # Amplitude Envelope (M)
            if (
                "median_amplitude_envelope" in indices_list
                or "acoustic_richness" in indices_list
            ):

                m_temp = maad.features.temporal_median(s)
                indices_temp["m"].append(m_temp)

            # Temporal Entropy (ht)
            if (
                "temporal_entropy" in indices_list
                or "acoustic_richness" in indices_list
            ):

                ht_temp = maad.features.temporal_entropy(s)
                indices_temp["ht"].append(ht_temp)

            if "temporal_leq" in indices_list:

                leq = maad.features.temporal_leq(s, fs, gain=42)
                indices_temp["leq"].append(leq)

            # Temporal Activity (ACT)
            if "temporal_activity" in indices_list:

                act_fract, act_count, act_mean = maad.features.temporal_activity(s, 6)

                indices_temp["act_fract"].append(act_fract)
                indices_temp["act_count"].append(act_count)
                indices_temp["act_mean"].append(act_mean)

            if "more_entropy" in indices_list:

                env = maad.sound.envelope(s)
                ht_havrda, ht_renyi, ht_paired_shannon, ht_gamma, ht_gini_simpson = (
                    maad.features.more_entropy(env**2, order=3)
                )

                indices_temp["ht_havrda"].append(ht_havrda)
                indices_temp["ht_renyi"].append(ht_renyi)
                indices_temp["ht_paired_shannon"].append(ht_paired_shannon)
                indices_temp["ht_gamma"].append(ht_gamma)
                indices_temp["ht_gini_simpson"].append(ht_gini_simpson)

            # Temporal Events (evn)
            if "temporal_events" in indices_list:

                evnt_fract, evn_mean, evn_count, _ = maad.features.temporal_events(
                    s, fs, 6
                )

                indices_temp["evnt_fract"].append(evnt_fract)
                indices_temp["evn_mean"].append(evn_mean)
                indices_temp["evn_count"].append(evn_count)

            # Acoustic Complexity Index (ACI)
            if (
                ("acoustic_complexity_index" in indices_list)
                or ("acoustic_diversity_index" in indices_list)
                or ("acoustic_eveness_index" in indices_list)
            ):
                sxx, tn, fn, ext = maad.sound.spectrogram(s, fs, mode="amplitude")

                if "acoustic_complexity_index" in indices_list:

                    aci_xx, aci_per_bin, aci_sum = (
                        maad.features.acoustic_complexity_index(sxx)
                    )
                    indices_temp["aci_sum"].append(aci_sum)
                    indices_temp["aci_xx"].append(aci_xx)
                    indices_temp["aci_per_bin"].append(aci_per_bin)

                if "acoustic_diversity_index" in indices_list:

                    adi = maad.features.acoustic_diversity_index(sxx, fn)
                    indices_temp["adi"].append(adi)

                if "acoustic_eveness_index" in indices_list:

                    aei = maad.features.acoustic_eveness_index(sxx, fn)
                    indices_temp["aei"].append(aei)

            if (
                ("frequency_entropy" in indices_list)
                or ("number_of_peaks" in indices_list)
                or ("spectral_activity" in indices_list)
                or ("spectral_events" in indices_list)
                or ("spectral_cover" in indices_list)
                or ("soundscape_index" in indices_list)
                or ("spectral_leq" in indices_list)
                or ("tfsd" in indices_list)
                or ("more_entropy" in indices_list)
                or ("acoustic_gradient_index" in indices_list)
                or ("frequency_raoq" in indices_list)
                or ("region_of_interest_index" in indices_list)
            ):

                sxx_power, tn, fn, ext = maad.sound.spectrogram(s, fs)

                # Frequency Entropy (hf)
                if "frequency_entropy" in indices_list:

                    hf, ht_per_bin = maad.features.frequency_entropy(sxx_power)
                    indices_temp["hf"].append(hf)
                    indices_temp["ht_per_bin"].append(ht_per_bin)

                # Number of Peaks (nb_peaks)
                if "number_of_peaks" in indices_list:

                    nb_peaks = maad.features.number_of_peaks(
                        sxx_power, fn, slopes=6, min_freq_dist=100
                    )
                    indices_temp["nb_peaks"].append(nb_peaks)

                if "tfsd" in indices_list:

                    tfsd = maad.features.tfsd(sxx_power, fn, tn)
                    indices_temp["tfsd"].append(tfsd)

                if "frequency_raoq" in indices_list:

                    s_power = maad.sound.avg_power_spectro(sxx_power)
                    raoq = maad.features.frequency_raoq(s_power, fn)
                    indices_temp["raoq"].append(raoq)

                if "soundscape_index" in indices_list:

                    ndsi, ratio_ba, antro_ph, bio_ph = maad.features.soundscape_index(
                        sxx_power, fn
                    )
                    indices_temp["ndsi"].append(ndsi)
                    indices_temp["ratio_ba"].append(ratio_ba)
                    indices_temp["antro_ph"].append(antro_ph)
                    indices_temp["bio_ph"].append(bio_ph)

                if "acoustic_gradient_index" in indices_list:

                    agi_xx, agi_per_bin, agi_mean, agi_sum = (
                        maad.features.acoustic_gradient_index(sxx_power, tn[1] - tn[0])
                    )

                    indices_temp["agi_xx"].append(agi_xx)
                    indices_temp["agi_per_bin"].append(agi_per_bin)
                    indices_temp["agi_mean"].append(agi_mean)
                    indices_temp["agi_sum"].append(agi_sum)

                if "spectral_leq" in indices_list:

                    leqf, leqf_per_bin = maad.features.spectral_leq(sxx_power, gain=42)
                    indices_temp["leqf"].append(leqf)
                    indices_temp["leqf_per_bin"].append(leqf_per_bin)

                if "more_entropy" in indices_list:

                    s_power = maad.sound.avg_power_spectro(sxx_power)
                    hf_havrda, hf_renyi, hf_paired_shannon, hf_gamma, hf_gini_simpson = (
                        maad.features.more_entropy(s_power, order=3)
                    )

                    indices_temp["hf_havrda"].append(hf_havrda)
                    indices_temp["hf_renyi"].append(hf_renyi)
                    indices_temp["hf_paired_shannon"].append(hf_paired_shannon)
                    indices_temp["hf_gamma"].append(hf_gamma)
                    indices_temp["hf_gini_simpson"].append(hf_gini_simpson)

                if "region_of_interest_index" in indices_list:

                    sxx_no_noise = maad.sound.median_equalizer(sxx_power)
                    sxx_db_no_noise = maad.util.power2dB(sxx_no_noise)
                    roi_total, roi_cover = maad.features.region_of_interest_index(
                        sxx_db_no_noise, tn, fn
                    )

                    indices_temp["roi_total"].append(roi_total)
                    indices_temp["roi_cover"].append(roi_cover)

                if (
                    ("spectral_activity" in indices_list)
                    or ("spectral_events" in indices_list)
                    or ("spectral_cover" in indices_list)
                ):

                    sxx_no_noise = maad.sound.median_equalizer(sxx_power, extent=ext)
                    sxx_db_no_noise = maad.util.power2dB(sxx_no_noise)

                    if "spectral_activity" in indices_list:

                        act_sp_fract_per_bin, act_sp_count_per_bin, act_sp_mean_per_bin = (
                            maad.features.spectral_activity(sxx_db_no_noise)
                        )

                        indices_temp["Meanact_sp_fract_per_bin"].append(
                            np.mean(act_sp_fract_per_bin)
                        )
                        indices_temp["Meanact_sp_count_per_bin"].append(
                            np.mean(act_sp_count_per_bin)
                        )
                        indices_temp["act_sp_fract_per_bin"].append(act_sp_fract_per_bin)
                        indices_temp["act_sp_count_per_bin"].append(act_sp_count_per_bin)
                        indices_temp["act_sp_mean_per_bin"].append(act_sp_mean_per_bin)

                    if "spectral_events" in indices_list:

                        (
                            evn_sp_fract_per_bin,
                            evn_sp_mean_per_bin,
                            evn_sp_count_per_bin,
                            evn_sp,
                        ) = maad.features.spectral_events(
                            sxx_db_no_noise,
                            dt=tn[1] - tn[0],
                            dB_threshold=6,
                            rejectDuration=0.1,
                            extent=ext,
                        )

                        mean_evn_sp_fract_per_bin = np.mean(evn_sp_fract_per_bin)

                        indices_temp["mean_evn_sp_fract_per_bin"].append(
                            mean_evn_sp_fract_per_bin
                        )
                        indices_temp["evn_sp_fract_per_bin"].append(evn_sp_fract_per_bin)
                        indices_temp["evn_sp_mean_per_bin"].append(evn_sp_mean_per_bin)
                        indices_temp["evn_sp_count_per_bin"].append(evn_sp_count_per_bin)
                        indices_temp["evn_sp"].append(evn_sp)

                    if "spectral_cover" in indices_list:

                        lfc, mfc, hfc = maad.features.spectral_cover(sxx_db_no_noise, fn)

                        indices_temp["lfc"].append(lfc)
                        indices_temp["mfc"].append(mfc)
                        indices_temp["hfc"].append(hfc)

            # Spectral Entropy (ACI)
            if "spectral_entropy" in indices_list:

                eas, ecu, ecv, eps, eps_kurt, eps_skew = maad.features.spectral_entropy(
                    sxx_power, fn
                )
                indices_temp["eas"].append(eas)
                indices_temp["ecu"].append(ecu)
                indices_temp["ecv"].append(ecv)
                indices_temp["eps"].append(eps)
                indices_temp["eps_kurt"].append(eps_kurt)
                indices_temp["eps_skew"].append(eps_skew)

        del s
        del fs

        gc.collect()

    # insert calculated indices into the original dataframe
    for key, value in indices_temp.items():
        df[key] = value

    # Acoustic Richness (AR)
    # This index uses the median of amplitude envelope andteporal entropy
    # to be calculated, and it is calculated out of the main loop
    if "acoustic_richness" in indices_list:
        ar_list = maad.features.acoustic_richness_index(df["ht"], df["m"])
        df["ar"] = ar_list

    # print(df.info())
    # print(sys.getsizeof(df)/1024)

    return df


# ----------------------------------------------------------------------------


def calculate_acoustic_indices_par_aux(args):
    return get_acoustic_indices(*args)


# ----------------------------------------------------------------------------


def parallelize_dataframe(df, func, args, num_cores) -> pd.DataFrame:

    # slices = math.floor(round(df.shape[0]/num_cores)/num_cores)*num_cores
    num_cores = mp.cpu_count()
    slices = max(1, min(df.shape[0], num_cores))

    # split the dataframe into num_cores parts
    df_split_temp = np.array_split(df, slices)

    df_split = []

    for i in range(slices):
        df_split.append([df_split_temp[i], args])

    # create a pool of processes
    with mp.Pool(num_cores) as pool:

        # apply the function on each part of the dataframe
        result = list(tqdm(pool.imap(func, df_split), total=len(df_split)))

        # close the pool and join the processes
        pool.close()
        pool.join()

    df = pd.concat(result)

    return df


# ----------------------------------------------------------------------------


def calculate_acoustic_indices(
    df_init,
    indices_list: list,
    store_df=False,
    base_dir=None,
    file_name=None,
    file_type=None,
    parallel=True,
) -> pd.DataFrame:
    """
        Calculate a set of acoustic indices for audio files and optionally
        store the results.

        This function calculates a set of specified acoustic indices for audio
        files in the input DataFrame. The supported indices include
        median_amplitude_envelope, temporal_entropy, acoustic_richness, and others.
        The calculated indices are added as columns to the DataFrame, and the
        resulting DataFrame is returned. If the 'store_df' parameter is set
        to True, the DataFrame can be saved to a file.

        Parameters
        ----------
        df_init : pd.DataFrame
            The DataFrame containing audio file data with a 'file_path'
            column specifying the path to each audio file.
        indices_list : List[str]
            A list of acoustic indices to calculate. Available indices are: 
            median_amplitude_envelope ,temporal_entropy ,acoustic_richness
            ,temporal_activity ,temporal_events ,acoustic_complexity_index
            ,frequency_entropy ,number_of_peaks ,spectral_entropy
            ,spectral_activity ,spectral_events ,spectral_cover
            ,soundscape_index ,acoustic_diversity_index ,acoustic_eveness_index
            ,temporal_leq ,spectral_leq ,tfsd ,more_entropy
            ,acoustic_gradient_index ,frequency_raoq
            ,region_of_interest_index
        store_df : bool, optional
            Whether to store the resulting DataFrame to a file. Default is False.
        base_dir : str, optional
            The base directory where the output file will be saved. Required if 'store_df' is True.
        file_name : str, optional
            The name of the output file to save the DataFrame. Required if 'store_df' is True.
        file_type : str, optional
            The file format for saving the DataFrame ('csv' or 'pickle').
            Required if 'store_df' is True.
        parallel : bool, optional
            Whether to use parallel processing for calculating indices. Default is True.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the calculated acoustic indices.

        Raises
        ------
        Exception
            If the selected indices are not available or if
            the DataFrame is missing the 'file_path' column.

        Examples
        --------
        >>> from maui import samples, acoustic_indices
        >>> df = samples.get_leec_audio_sample()
        >>> indices_list = ['median_amplitude_envelope', 'temporal_entropy']
        >>> df = acoustic_indices.calculate_acoustic_indices(df, indices_list, parallel=False)

        Notes
        -----
        - The function calculates a specified set of acoustic
            indices for each audio file in the DataFrame.
        - The 'indices_list' parameter should contain a list of available
            indices to be calculated.
        - The 'store_df', 'base_dir', 'file_name', and 'file_type' parameters
            control whether and where to save the resulting DataFrame.
        - If parallel processing is enabled ('parallel' is True),
            the function will use multiple CPU cores for faster calculations.

    """

    # check if the selected indices are available
    available_indices = [
        "median_amplitude_envelope",
        "temporal_entropy",
        "acoustic_richness",
        "temporal_activity",
        "temporal_events",
        "acoustic_complexity_index",
        "frequency_entropy",
        "number_of_peaks",
        "spectral_entropy",
        "spectral_activity",
        "spectral_events",
        "spectral_cover",
        "soundscape_index",
        "acoustic_diversity_index",
        "acoustic_eveness_index",
        "temporal_leq",
        "spectral_leq",
        "tfsd",
        "more_entropy",
        "acoustic_gradient_index",
        "frequency_raoq",
        "region_of_interest_index",
    ]

    diff = set(indices_list).difference(available_indices)
    if not set(indices_list).issubset(available_indices):
        raise Exception(
            f"""The indices {diff} are not available. """\
            f"""The list of available indices is: {available_indices}"""
        )

    # check if the dataframe contains the correct columns
    necessary_colums = ["file_path"]

    if not set(necessary_colums).issubset(df_init.columns):
        raise Exception(
            "Your dataset must contain a column named 'file_path', "\
            "which shows the path for each audio file."
        )

    args = {
        "indices_list": indices_list,
        "total_iterations": df_init.shape[0],
        "parallel": parallel,
    }

    if parallel:
        df_processed = parallelize_dataframe(
            df_init, calculate_acoustic_indices_par_aux, args, mp.cpu_count()
        )
    else:
        df_processed = get_acoustic_indices(df_init, args)

    if store_df:
        maui.io.store_df(df_processed, file_type, base_dir, file_name)

    return df_processed
