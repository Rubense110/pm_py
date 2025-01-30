from pm4py.objects.log.importer.xes import importer as xes_importer
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pm')))
import common

def calculate_log_metrics(log_path):
    """
    Log metrics obtained from the paper
    On making informed decisions regarding clustering event logs in process discovery
    """
    log = xes_importer.apply(log_path)

    ## Traces
    count_traces = len(log)
    count_events = sum(len(trace) for trace in log)
    unique_events = set(event['concept:name'] for trace in log for event in trace)
    count_unique_events = len(unique_events)
    trace_len = count_events / count_traces
    variants = Counter(tuple(event['concept:name'] for event in trace) for trace in log)
    most_common_variant_count = variants.most_common(1)[0][1]
    ratio_most_common_variant = most_common_variant_count / count_traces
    ratio_top1_variants = ratio_most_common_variant
    top5_variants_count = sum(count for variant, count in variants.most_common(5))
    ratio_top5_variants = top5_variants_count / count_traces
    top10_variants_count = sum(count for variant, count in variants.most_common(10))
    ratio_top10_variants = top10_variants_count / count_traces
    top20_variants_count = sum(count for variant, count in variants.most_common(20))
    ratio_top20_variants = top20_variants_count / count_traces
    top50_variants_count = sum(count for variant, count in variants.most_common(50))
    ratio_top50_variants = top50_variants_count / count_traces
    top75_variants_count = sum(count for variant, count in variants.most_common(75))
    ratio_top75_variants = top75_variants_count / count_traces

    ## Tasks

    ##DEBUG
    print(f"count-events: {count_events}")
    print(f"count-traces: {count_traces}")
    print(f"count-unique-events: {count_unique_events}")
    print(f"trace-len: {trace_len}")
    print("variant-ocurrence:")
    for variant, count in variants.items():
        print(f"    {' -> '.join(variant)}: {count}")
    print(f"ratio-most-common-variant: {ratio_most_common_variant}")
    print(f"ratio-top1-variants: {ratio_top1_variants}")
    print(f"ratio-top5-variants: {ratio_top5_variants}")
    print(f"ratio-top10-variants: {ratio_top10_variants}")
    print(f"ratio-top20-variants: {ratio_top20_variants}")
    print(f"ratio-top50-variants: {ratio_top50_variants}")
    print(f"ratio-top75-variants: {ratio_top75_variants}")

if __name__ == "__main__":
    log = common.log_closed[1]
    calculate_log_metrics(log)