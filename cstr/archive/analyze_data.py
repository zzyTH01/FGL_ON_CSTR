import pickle, numpy as np

with open('cstr/data.pkl', 'rb') as f:
    temp_data = pickle.load(f)
with open('cstr/data_h2o.pkl', 'rb') as f:
    h2o_data = pickle.load(f)

temp = temp_data[:,0].numpy()
h2o = h2o_data[:,0].numpy()
time = np.arange(len(temp)) * 0.1

# ========== Single Oscillation Cycle (t=100~130s) ==========
print('=== Single Oscillation Cycle (t=100~130s) ===')
print(f'{"Time(s)":>8s}  {"Temp(K)":>10s}  {"H2O":>10s}  {"Phase":>20s}')
print('-' * 56)
mask = (time >= 100) & (time <= 130)
indices = np.where(mask)[0]
for i in indices[::10]:
    phase = ''
    if temp[i] > 1000:
        phase = 'IGNITION'
    elif h2o[i] > 0.5:
        phase = 'H2O peak'
    elif h2o[i] < 0.01:
        phase = 'H2O ~0'
    elif temp[i] < 780:
        phase = 'Baseline'
    print(f'{time[i]:8.1f}  {temp[i]:10.1f}  {h2o[i]:10.6f}  {phase:>20s}')

# ========== Key events in first 60s ==========
print('\n=== Oscillation Dynamics (first 60s, key events) ===')
mask = time <= 60
indices = np.where(mask)[0]
in_ignition = False
last_state = 'baseline'
for i in indices:
    if temp[i] > 800 and not in_ignition:
        in_ignition = True
        print(f't={time[i]:5.1f}s: IGNITION starts   T={temp[i]:6.1f}K  H2O={h2o[i]:.4f}')
    elif temp[i] < 780 and in_ignition:
        in_ignition = False
        print(f't={time[i]:5.1f}s: Ignition ends    T={temp[i]:6.1f}K  H2O={h2o[i]:.4f}')
    elif h2o[i] > 0.9 and last_state != 'peak':
        last_state = 'peak'
        print(f't={time[i]:5.1f}s: H2O max           T={temp[i]:6.1f}K  H2O={h2o[i]:.4f}')
    elif h2o[i] < 0.001 and last_state == 'draining':
        last_state = 'exhausted'
        print(f't={time[i]:5.1f}s: H2O ~0            T={temp[i]:6.1f}K  H2O={h2o[i]:.6f}')
    elif h2o[i] < 0.01 and last_state == 'peak':
        last_state = 'draining'

# ========== Temperature spike characterization ==========
print('\n=== Temperature Spike Events ===')
spike_mask = temp > 1000
spike_diff = np.diff(np.concatenate([[0], spike_mask.astype(int), [0]]))
spike_starts = np.where(spike_diff == 1)[0]
spike_ends = np.where(spike_diff == -1)[0]
print(f'Number of spikes: {len(spike_starts)}')
for j, (s, e) in enumerate(zip(spike_starts[:5], spike_ends[:5])):
    duration = (e - s) * 0.1
    max_t = temp[s:e].max()
    rise = max_t - temp[s]
    print(f'  Spike {j+1}: t=[{time[s]:.1f}s, {time[e-1]:.1f}s]  '
          f'dur={duration:.1f}s  peak={max_t:.0f}K  rise={rise:.0f}K  ')

# ========== H2O phase analysis ==========
print('\n=== H2O Mass Fraction Phase Analysis ===')
# Rising phase: from ~0 to peak (>0.9)
# Plateau phase: stays high for a while
# Falling phase: from peak back to ~0
# Let's analyze one complete cycle
# Find cycle start (H2O crosses above 0.01 after being < 0.01)
above = h2o > 0.01
crossings = np.diff(np.concatenate([[0], above.astype(int), [0]]))
rise_starts = np.where(crossings == 1)[0]  # H2O rising above 0.01
print(f'H2O rise events (>0.01): {len(rise_starts)}')
if len(rise_starts) >= 2:
    cycle_len = (rise_starts[2] - rise_starts[1]) * 0.1
    print(f'Typical cycle length: {cycle_len:.1f}s ({cycle_len/0.1:.0f} steps)')

# ========== Sliding window analysis ==========
print('\n=== Sliding Window Analysis (lookback=8, H=5) ===')
L, H = 8, 5
print(f'With lookback={L}, H={H}:')
print(f'  Input window: {L} steps = {L*0.1:.1f}s of history')
print(f'  Target: {H} steps ahead = {H*0.1:.1f}s into future')
print(f'  Total span: {(L+H)*0.1:.1f}s')

# Compare with oscillation period
cycle_steps = 71.5  # ~7.15s / 0.1
print(f'\n  Oscillation period: ~{cycle_steps:.0f} steps = ~7.15s')
print(f'  Lookback covers: {L/cycle_steps*100:.0f}% of a cycle')
print(f'  H-step gap covers: {H/cycle_steps*100:.0f}% of a cycle')
print(f'  Total span covers: {(L+H)/cycle_steps*100:.0f}% of a cycle')

# ========== Temperature vs H2O correlation ==========
print('\n=== Temperature vs H2O Cross-Correlation ===')
from scipy import signal
corr = signal.correlate(temp - temp.mean(), h2o - h2o.mean(), mode='same')
lags = signal.correlation_lags(len(temp), len(h2o), mode='same')
max_lag = lags[np.argmax(np.abs(corr))]
print(f'Max correlation at lag: {max_lag} steps ({max_lag*0.1:.1f}s)')
print(f'Temperature leads H2O by {-max_lag*0.1:.1f}s' if max_lag < 0 else f'H2O leads Temperature by {max_lag*0.1:.1f}s')

# ========== Discretization analysis ==========
print('\n=== Discretization Analysis (50 bins, H2O) ===')
bin_edges = np.linspace(h2o.min(), h2o.max(), 49)
bin_counts = np.digitize(h2o, bin_edges)
# Count samples per bin
unique, counts = np.unique(bin_counts, return_counts=True)
zero_bin_count = counts[0] if 0 in unique else 0
print(f'Bin 0 (near-zero) samples: {zero_bin_count} ({zero_bin_count/len(h2o)*100:.1f}%)')
print(f'Bins with <10 samples: {sum(counts < 10)}/{len(counts)}')
print(f'Max bin count: {counts.max()} (bin {unique[np.argmax(counts)]})')
print(f'Imbalanced bins: {sum(counts < len(h2o)*0.001)} bins have <0.1% of data')
