import pm4py
import pandas as pd
import numpy as np
from scipy import stats


def extract_log_features(log_path):
    """
    Extrae características de un log XES
    
    Parámetros:
    log_path (str): Ruta del archivo XES
    
    Devuelve:
    dict: Diccionario con características extraídas
    """

    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)
    
    
    def trace_level_features(log):
        lengths = [len(trace) for trace in log]
        return {
            'trace_length_min': np.min(lengths),
            'trace_length_max': np.max(lengths),
            'trace_length_mean': np.mean(lengths),
            'trace_length_median': np.median(lengths),
            'trace_length_std': np.std(lengths),
            'trace_length_variance': np.var(lengths),
            'trace_length_percentile_25': np.percentile(lengths, 25),
            'trace_length_percentile_75': np.percentile(lengths, 75),
            'trace_length_entropy': stats.entropy(lengths)
        }
    
    def trace_variant_features(log):

        variants = pm4py.get_variants(log)
        variant_frequencies = list(variants.values())
        
        return {
            'variant_count': len(variants),
            'variant_mean_count': np.mean(variant_frequencies),
            'variant_std_count': np.std(variant_frequencies),
            'top_variant_ratio': max(variant_frequencies) / len(log),
            'top_1_percent_ratio': sorted(variant_frequencies, reverse=True)[:int(len(log)*0.01)],
            'top_5_percent_ratio': sorted(variant_frequencies, reverse=True)[:int(len(log)*0.05)]
        }
    
    def activity_features(df):

        activities = df['concept:name'].unique()
        activity_freq = df['concept:name'].value_counts()
        
        return {
            'total_activities': len(activities),
            'unique_activities': len(set(activities)),
            'most_frequent_activity': activity_freq.index[0],
            'most_frequent_activity_count': activity_freq.iloc[0],
            'least_frequent_activity': activity_freq.index[-1],
            'least_frequent_activity_count': activity_freq.iloc[-1]
        }
    
    def log_level_features(log, df):
        return {
            'total_traces': len(log),
            'total_events': len(df),
            'traces_to_events_ratio': len(log) / len(df)
        }
    
    features = {
        **trace_level_features(log),
        **trace_variant_features(log),
        **activity_features(df),
        **log_level_features(log, df)
    }
    
    return features

if __name__ == "__main__":
    log_path = 'event_logs/Financial/BPI_Challenge_2012.xes'
    features = extract_log_features(log_path)
    print(features)