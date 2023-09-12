import numpy as np
import pandas as pd
import math

import gc

from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import sys

import maad

import maui.io




def get_acoustic_indices(df, args: dict):

    
    indices_list = args['indices_list']
    
    # create temp lit of indices to keep appending for each row
    indices_temp = {}
    
    for ind in indices_list:
        if ind ==  'median_amplitude_envelope':
            indices_temp['M'] = []
        if ind ==  'temporal_entropy':
            indices_temp['Ht'] = []
        if ind ==  'temporal_activity':
            indices_temp['ACTfract'] = []
            indices_temp['ACTcount'] = []
            indices_temp['ACTmean'] = []
        if ind ==  'temporal_events':
            indices_temp['EVNtFract'] = []
            indices_temp['EVNmean'] = []
            indices_temp['EVNcount'] = []
        if ind ==  'frequency_entropy':
            indices_temp['Hf'] = []
            indices_temp['Ht_per_bin'] = []
        if ind ==  'number_of_peaks':
            indices_temp['NBPeaks'] = []
        if ind ==  'spectral_entropy':
            indices_temp['EAS'] = []
            indices_temp['ECU'] = []
            indices_temp['ECV'] = []
            indices_temp['EPS'] = []
            indices_temp['EPS_KURT'] = []
            indices_temp['EPS_SKEW'] = []
        if ind ==  'spectral_activity':
            indices_temp['ACTspfract_per_bin'] = []
            indices_temp['ACTspcount_per_bin'] = []
            indices_temp['ACTspmean_per_bin'] = []
            indices_temp['MeanACTspfract_per_bin'] = []
            indices_temp['MeanACTspcount_per_bin'] = []
        if ind ==  'spectral_events':
            indices_temp['MeanEVNspFract_per_bin'] = []
            indices_temp['EVNspFract_per_bin'] = []
            indices_temp['EVNspMean_per_bin'] = []
            indices_temp['EVNspCount_per_bin'] = []
            indices_temp['EVNsp'] = []
        if ind ==  'spectral_cover':
            indices_temp['LFC'] = []
            indices_temp['MFC'] = []
            indices_temp['HFC'] = []
        if ind ==  'soundscape_index':
            indices_temp['NDSI'] = []
            indices_temp['ratioBA'] = []
            indices_temp['antroPh'] = []
            indices_temp['bioPh'] = []
        if ind ==  'acoustic_diversity_index':
            indices_temp['ADI'] = []
        if ind ==  'acoustic_eveness_index':
            indices_temp['AEI'] = []
        if ind ==  'temporal_leq':
            indices_temp['Leq'] = []
        if ind ==  'spectral_leq':
            indices_temp['Leqf'] = []
            indices_temp['Leqf_per_bin'] = []
        if ind ==  'tfsd':
            indices_temp['tfsd'] = []
        if ind ==  'more_entropy':
            indices_temp['Ht_Havrda'] = []
            indices_temp['Ht_Renyi'] = []
            indices_temp['Ht_pairedShannon'] = []
            indices_temp['Ht_gamma'] = []
            indices_temp['Ht_GiniSimpson'] = []
            indices_temp['Hf_Havrda'] = []
            indices_temp['Hf_Renyi'] = []
            indices_temp['Hf_pairedShannon'] = []
            indices_temp['Hf_gamma'] = []
            indices_temp['Hf_GiniSimpson'] = []
        if ind ==  'acoustic_gradient_index':
            indices_temp['AGI_xx'] = []
            indices_temp['AGI_per_bin'] = []
            indices_temp['AGI_mean'] = []
            indices_temp['AGI_sum'] = []
        if ind ==  'frequency_raoq':
            indices_temp['RAOQ'] = []
        if ind ==  'region_of_interest_index':
            indices_temp['ROItotal'] = []
            indices_temp['ROIcover'] = []
        if ind ==  'acoustic_complexity_index':
            indices_temp['ACI_sum'] = []
            indices_temp['ACI_xx'] = []
            indices_temp['ACI_per_bin'] = []
        

            
    
    # for each row of the dataframe, a list of acoustic indices (given by indices_list parameter) is calculated 
    iterator_df = range(df.shape[0]) if args['parallel'] else tqdm(range(df.shape[0]))

    for index in iterator_df:
#     for index, row in df.iterrows(): 
        row = df.iloc[index]
        
        # load audio file
        s, fs = maad.sound.load(row['file_path'])
        if (len(s) == 0):
            for key in indices_temp:
                indices_temp[key].append(None)
                
        else:
            
        
            # Amplitude Envelope (M)
            if ('median_amplitude_envelope' in indices_list or 'acoustic_richness' in indices_list):

                M_temp = maad.features.temporal_median(s)
                indices_temp['M'].append(M_temp)

            # Temporal Entropy (Ht)
            if ('temporal_entropy' in indices_list or 'acoustic_richness' in indices_list):

                Ht_temp = maad.features.temporal_entropy(s)
                indices_temp['Ht'].append(Ht_temp)


            if ('temporal_leq' in indices_list):

                Leq = maad.features.temporal_leq (s, fs, gain=42)
                indices_temp['Leq'].append(Leq)

            # Temporal Activity (ACT)
            if ('temporal_activity' in indices_list):

                ACTfract, ACTcount, ACTmean = maad.features.temporal_activity(s, 6)


                indices_temp['ACTfract'].append(ACTfract)
                indices_temp['ACTcount'].append(ACTcount)
                indices_temp['ACTmean'].append(ACTmean)

            if ('more_entropy' in indices_list):

                env = maad.sound.envelope(s)
                Ht_Havrda, Ht_Renyi, Ht_pairedShannon, Ht_gamma, Ht_GiniSimpson = maad.features.more_entropy(env**2, order=3)


                indices_temp['Ht_Havrda'].append(Ht_Havrda)
                indices_temp['Ht_Renyi'].append(Ht_Renyi)
                indices_temp['Ht_pairedShannon'].append(Ht_pairedShannon)
                indices_temp['Ht_gamma'].append(Ht_gamma)
                indices_temp['Ht_GiniSimpson'].append(Ht_GiniSimpson)


            # Temporal Events (EVN)
            if ('temporal_events' in indices_list):

                EVNtFract, EVNmean, EVNcount, EVN = maad.features.temporal_events (s, fs, 6)


                indices_temp['EVNtFract'].append(EVNtFract)
                indices_temp['EVNmean'].append(EVNmean)
                indices_temp['EVNcount'].append(EVNcount)

            # Acoustic Complexity Index (ACI)
            if (('acoustic_complexity_index' in indices_list) or ('acoustic_diversity_index' in indices_list) or ('acoustic_eveness_index' in indices_list)):
                Sxx, tn, fn, ext = maad.sound.spectrogram (s, fs, mode='amplitude') 

                if ('acoustic_complexity_index' in indices_list):


                    ACI_xx, ACI_per_bin, ACI_sum  = maad.features.acoustic_complexity_index(Sxx)
                    indices_temp['ACI_sum'].append(ACI_sum)
                    indices_temp['ACI_xx'].append(ACI_xx)
                    indices_temp['ACI_per_bin'].append(ACI_per_bin)

                if ('acoustic_diversity_index' in indices_list):

                    ADI  = maad.features.acoustic_diversity_index(Sxx,fn)        
                    indices_temp['ADI'].append(ADI)


                if ('acoustic_eveness_index' in indices_list):

                    AEI  = maad.features.acoustic_eveness_index(Sxx,fn)        
                    indices_temp['AEI'].append(AEI)



            if (('frequency_entropy' in indices_list) or ('number_of_peaks' in indices_list) or ('spectral_activity' in indices_list) or 
                ('spectral_events' in indices_list) or ('spectral_cover' in indices_list) or ('soundscape_index' in indices_list) or 
                ('spectral_leq' in indices_list)  or ('tfsd' in indices_list)  or ('more_entropy' in indices_list)  or 
                ('acoustic_gradient_index' in indices_list)  or ('frequency_raoq' in indices_list)  or ('region_of_interest_index' in indices_list)  ):

                Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs) 

                # Frequency Entropy (Hf)
                if ('frequency_entropy' in indices_list):

                    Hf, Ht_per_bin = maad.features.frequency_entropy(Sxx_power)
                    indices_temp['Hf'].append(Hf)
                    indices_temp['Ht_per_bin'].append(Ht_per_bin)


                # Number of Peaks (NBPeaks)
                if ('number_of_peaks' in indices_list): 

                    NBPeaks = maad.features.number_of_peaks(Sxx_power, fn, slopes=6, min_freq_dist=100) 
                    indices_temp['NBPeaks'].append(NBPeaks)

                if ('tfsd' in indices_list): 

                    tfsd = maad.features.tfsd(Sxx_power,fn, tn) 
                    indices_temp['tfsd'].append(tfsd)

                if ('frequency_raoq' in indices_list): 

                    S_power = maad.sound.avg_power_spectro(Sxx_power) 
                    RAOQ = maad.features.frequency_raoq(S_power, fn)
                    indices_temp['RAOQ'].append(RAOQ)


                if ('soundscape_index' in indices_list): 

                    NDSI, ratioBA, antroPh, bioPh  = maad.features.soundscape_index(Sxx_power,fn) 
                    indices_temp['NDSI'].append(NDSI)
                    indices_temp['ratioBA'].append(ratioBA)
                    indices_temp['antroPh'].append(antroPh)
                    indices_temp['bioPh'].append(bioPh)

                if ('acoustic_gradient_index' in indices_list): 

                    AGI_xx, AGI_per_bin, AGI_mean, AGI_sum = maad.features.acoustic_gradient_index(Sxx_power,tn[1]-tn[0])

                    indices_temp['AGI_xx'].append(AGI_xx)
                    indices_temp['AGI_per_bin'].append(AGI_per_bin)
                    indices_temp['AGI_mean'].append(AGI_mean)
                    indices_temp['AGI_sum'].append(AGI_sum)

                if ('spectral_leq' in indices_list): 

                    Leqf, Leqf_per_bin = maad.features.spectral_leq(Sxx_power, gain=42)
                    indices_temp['Leqf'].append(Leqf)
                    indices_temp['Leqf_per_bin'].append(Leqf_per_bin)

                if ('more_entropy' in indices_list): 

                    S_power = maad.sound.avg_power_spectro(Sxx_power)
                    Hf_Havrda, Hf_Renyi, Hf_pairedShannon, Hf_gamma, Hf_GiniSimpson = maad.features.more_entropy(S_power, order=3)

                    indices_temp['Hf_Havrda'].append(Hf_Havrda)
                    indices_temp['Hf_Renyi'].append(Hf_Renyi)
                    indices_temp['Hf_pairedShannon'].append(Hf_pairedShannon)
                    indices_temp['Hf_gamma'].append(Hf_gamma)
                    indices_temp['Hf_GiniSimpson'].append(Hf_GiniSimpson)

                if ('region_of_interest_index' in indices_list): 

                    Sxx_noNoise= maad.sound.median_equalizer(Sxx_power) 
                    Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
                    ROItotal, ROIcover = maad.features.region_of_interest_index(Sxx_dB_noNoise, tn, fn)

                    indices_temp['ROItotal'].append(ROItotal)
                    indices_temp['ROIcover'].append(ROIcover)



                if (('spectral_activity' in indices_list) or ('spectral_events' in indices_list) or ('spectral_cover' in indices_list)  ):
    #                 print('entrou')

                    Sxx_noNoise= maad.sound.median_equalizer(Sxx_power, extent=ext) 
                    Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)

                    if ('spectral_activity' in indices_list):


                        ACTspfract_per_bin, ACTspcount_per_bin, ACTspmean_per_bin = maad.features.spectral_activity(Sxx_dB_noNoise)  

                        indices_temp['MeanACTspfract_per_bin'].append(np.mean(ACTspfract_per_bin))
                        indices_temp['MeanACTspcount_per_bin'].append(np.mean(ACTspcount_per_bin))
                        indices_temp['ACTspfract_per_bin'].append(ACTspfract_per_bin)
                        indices_temp['ACTspcount_per_bin'].append(ACTspcount_per_bin)
                        indices_temp['ACTspmean_per_bin'].append(ACTspmean_per_bin)

                    if ('spectral_events' in indices_list):

                        EVNspFract_per_bin, EVNspMean_per_bin, EVNspCount_per_bin, EVNsp = maad.features.spectral_events(Sxx_dB_noNoise, 
                                                                                                                         dt=tn[1]-tn[0], dB_threshold=6, 
                                                                                                                         rejectDuration=0.1, extent=ext)  

                        MeanEVNspFract_per_bin = np.mean(EVNspFract_per_bin)

                        indices_temp['MeanEVNspFract_per_bin'].append(MeanEVNspFract_per_bin)
                        indices_temp['EVNspFract_per_bin'].append(EVNspFract_per_bin)
                        indices_temp['EVNspMean_per_bin'].append(EVNspMean_per_bin)
                        indices_temp['EVNspCount_per_bin'].append(EVNspCount_per_bin)
                        indices_temp['EVNsp'].append(EVNsp)

                    if ('spectral_cover' in indices_list):


                        LFC, MFC, HFC = maad.features.spectral_cover(Sxx_dB_noNoise, fn)  

                        indices_temp['LFC'].append(LFC)
                        indices_temp['MFC'].append(MFC)
                        indices_temp['HFC'].append(HFC)




            # Spectral Entropy (ACI)
            if ('spectral_entropy' in indices_list):

                EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = maad.features.spectral_entropy(Sxx_power, fn) 
                indices_temp['EAS'].append(EAS)
                indices_temp['ECU'].append(ECU)
                indices_temp['ECV'].append(ECV)
                indices_temp['EPS'].append(EPS)
                indices_temp['EPS_KURT'].append(EPS_KURT)
                indices_temp['EPS_SKEW'].append(EPS_SKEW)
        
        del(s)
        del(fs)
        gc.collect()


    
    # insert calculated indices into the original dataframe
    for key in indices_temp:
        df[key] = indices_temp[key]
    
    # Acoustic Richness (AR)
    # This index uses the median of amplitude envelope andteporal entropy to be calculated, and it is calculated out of the main loop
    if (('acoustic_richness' in indices_list)):
        AR_list = maad.features.acoustic_richness_index(df['Ht'], df['M'])
        df['AR'] = AR_list
    
    # print(df.info())
    # print(sys.getsizeof(df)/1024)
    
    
    return df
    
#-----------------------------------------------------------------------------------------------------------------------------------


def calculate_acoustic_indices_par_aux(args):
    return get_acoustic_indices(*args)


#-----------------------------------------------------------------------------------------------------------------------------------


def parallelize_dataframe(df, func, args, num_cores):
    
    # slices = math.floor(round(df.shape[0]/num_cores)/num_cores)*num_cores
    slices = math.floor(round(df.shape[0]*df.shape[1] / 8)/8)
    
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

#-----------------------------------------------------------------------------------------------------------------------------------



def calculate_acoustic_indices(df_init, indices_list: list, store_df=False, base_dir=None, file_name=None, file_type=None, parallel=True):
    """
    Calculate a set of acoustic indices for audio files and optionally store the results.

    This function calculates a set of specified acoustic indices for audio files in the input DataFrame. The supported
    indices include median_amplitude_envelope, temporal_entropy, acoustic_richness, and others. The calculated indices
    are added as columns to the DataFrame, and the resulting DataFrame is returned. If the 'store_df' parameter is set
    to True, the DataFrame can be saved to a file.

    Parameters
    ----------
    df_init : pd.DataFrame
        The DataFrame containing audio file data with a 'file_path' column specifying the path to each audio file.
    indices_list : List[str]
        A list of acoustic indices to calculate.
    store_df : bool, optional
        Whether to store the resulting DataFrame to a file. Default is False.
    base_dir : str, optional
        The base directory where the output file will be saved. Required if 'store_df' is True.
    file_name : str, optional
        The name of the output file to save the DataFrame. Required if 'store_df' is True.
    file_type : str, optional
        The file format for saving the DataFrame, e.g., 'csv', 'xlsx'. Required if 'store_df' is True.
    parallel : bool, optional
        Whether to use parallel processing for calculating indices. Default is True.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the calculated acoustic indices.

    Raises
    ------
    Exception
        If the selected indices are not available or if the DataFrame is missing the 'file_path' column.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv('audio_data.csv')
    >>> indices_list = ['median_amplitude_envelope', 'temporal_entropy']
    >>> result_df = calculate_acoustic_indices(df, indices_list, store_df=True, base_dir='output', file_name='indices_result.csv', file_type='csv', parallel=True)

    Notes
    -----
    - The function calculates a specified set of acoustic indices for each audio file in the DataFrame.
    - The 'indices_list' parameter should contain a list of available indices to be calculated.
    - The 'store_df', 'base_dir', 'file_name', and 'file_type' parameters control whether and where to save the resulting DataFrame.
    - If parallel processing is enabled ('parallel' is True), the function will use multiple CPU cores for faster calculations.
    """


    # check if the selected indices are available
    available_indices = ['median_amplitude_envelope', 'temporal_entropy', 'acoustic_richness', 'temporal_activity', 'temporal_events', 'acoustic_complexity_index', 
                         'frequency_entropy', 'number_of_peaks', 'spectral_entropy', 'spectral_activity', 'spectral_events', 'spectral_cover', 'soundscape_index',
                         'acoustic_diversity_index', 'acoustic_eveness_index', 'temporal_leq', 'spectral_leq', 'tfsd', 'more_entropy', 'acoustic_gradient_index',
                         'frequency_raoq', 'region_of_interest_index']
    
    diff = set(indices_list).difference(available_indices)
    if not (set(indices_list).issubset(available_indices)): raise Exception (f'''The indices {diff} are not available. The list of available indices is: {available_indices}''')
        
    # check if the dataframe contains the correct columns
    necessary_colums = ['file_path']
    
    if not (set(necessary_colums).issubset(df_init.columns)): raise Exception("Your dataset must contain a column named 'file_path', which shows the path for each audio file.")
    
    args = {'indices_list':indices_list, 'total_iterations': df_init.shape[0], 'parallel':parallel}

    if parallel:
        df_processed = parallelize_dataframe(df_init, calculate_acoustic_indices_par_aux, args, mp.cpu_count())
    else: 
        df_processed = get_acoustic_indices(df_init, args)


    if (store_df):
        maui.io.store_df(df_processed, file_type, base_dir, file_name)
    
    return df_processed