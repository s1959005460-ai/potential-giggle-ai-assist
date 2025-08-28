import streamlit as st
import time, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='FedGNN Dashboard', layout='wide')

st.title('FedGNN Training Dashboard')

metrics_file = os.path.join('logs', 'metrics.jsonl')

st.sidebar.header('Controls')
auto_refresh = st.sidebar.checkbox('Auto refresh', value=True)
refresh_interval = st.sidebar.slider('Refresh (s)', 1, 10, 3)

st.header('Global metrics')
acc_chart = st.empty()
loss_chart = st.empty()

st.header('Client statistics (last round)')
clients_tbl = st.empty()

st.header('Selected clients history')
sel_hist = st.empty()

st.header('Privacy & Alerts')
privacy_box = st.empty()
alerts_box = st.empty()

def read_metrics():
    if not os.path.exists(metrics_file):
        return []
    lines = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                lines.append(json.loads(line.strip()))
            except:
                continue
    return lines

def render():
    data = read_metrics()
    if len(data) == 0:
        st.info('No metrics yet. Run run_experiment.py to produce metrics.')
        return
    rounds = [d['round'] for d in data]
    accs = [d['global']['acc'] for d in data]
    losses = [d['global']['loss'] for d in data]
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(rounds, accs, marker='o')
    ax[0].set_title('Global Accuracy')
    ax[0].set_xlabel('Round')
    ax[0].set_ylabel('Accuracy')
    ax[1].plot(rounds, losses, marker='o')
    ax[1].set_title('Global Loss')
    ax[1].set_xlabel('Round')
    acc_chart.pyplot(fig)

    last = data[-1]
    client_ids = list(range(len(last.get('client_counts', []))))
    df = pd.DataFrame({
        'client_id': client_ids,
        'samples': last.get('client_counts', []),
        'update_norm': last.get('client_norms', [])
    })
    clients_tbl.dataframe(df)

    sel = []
    for d in data:
        sel.append({'round': d['round'], 'selected': d['selected_clients']})
    sel_hist.write(sel)

    # privacy placeholder (if federated_privacy used, you can write epsilon to logs)
    privacy_box.write('Privacy metrics not enabled in demo. Use federated_privacy module to compute epsilon.')

    alerts_box.write('No alerts yet. If robust aggregation flags malicious clients, they will appear here.')

if auto_refresh:
    while True:
        render()
        time.sleep(refresh_interval)
else:
    render()
