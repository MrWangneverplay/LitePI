import os, re, glob, statistics, csv

BENCH = '/root/delphi/run/bench18'
OUT = '/mnt/d/王梓杨/项目组/wzy/results/E_delphi/delphi_minionn_18thread.csv'
DUR = re.compile(r'^(?:.*?)End:\s+Client (offline|online) phase\.+([0-9.]+)\s*(ms|s|µs|us|m)$')
COMM = re.compile(r'(Read|Wrote) ([0-9]+) bytes')

def parse_log(path):
    off = on = None; sec = None; data = {}
    for line in open(path, errors='replace'):
        m = DUR.match(line)
        if m:
            v = float(m.group(2)); u = m.group(3)
            v = v/1000 if u == 'ms' else (v/1e6 if u in ('µs','us') else (v*60 if u == 'm' else v))
            if m.group(1) == 'offline': off = v
            else: on = v
        if 'Offline Communication' in line: sec = 'off'; data.setdefault(sec, [0,0])
        elif 'Online Communication' in line: sec = 'on'; data.setdefault(sec, [0,0])
        else:
            c = COMM.search(line)
            if c and sec:
                data[sec][0 if c.group(1) == 'Read' else 1] = int(c.group(2))
    return off, on, *data.get('off', [0,0]), *data.get('on', [0,0])

def peak_rss(path):
    for line in open(path, errors='replace'):
        if 'Maximum resident set size' in line:
            return int(line.split(':')[1].strip())
    return 0

rows = []
for p in sorted(glob.glob(os.path.join(BENCH, 'client_L*_r*.log'))):
    m = re.match(r'client_L(\d+)_r(\d+)\.log', os.path.basename(p))
    L, r = int(m.group(1)), int(m.group(2))
    off, on, o_rd, o_wr, n_rd, n_wr = parse_log(p)
    crss = peak_rss(p[:-4] + '.time')
    srss = peak_rss(os.path.join(BENCH, f'server_L{L}_r{r}.time'))
    rows.append([L, r, off, on, o_rd + o_wr, n_rd + n_wr, crss, srss])

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layers','run','offline_s','online_s','comm_off_bytes','comm_on_bytes','client_rss_kb','server_rss_kb'])
    w.writerows(rows)

print('18-thread results (3 runs each):')
for L in sorted({r[0] for r in rows}):
    g = [r for r in rows if r[0] == L]
    print('layers=%d offline=%.3fs±%.3f online=%.3fs±%.3f comm_off=%.1fMiB comm_on=%.1fMiB rss=%.0fMiB/%.0fMiB'
          % (L, statistics.mean(r[2] for r in g), statistics.pstdev(r[2] for r in g),
             statistics.mean(r[3] for r in g), statistics.pstdev(r[3] for r in g),
             statistics.mean(r[4] for r in g) / 1048576, statistics.mean(r[5] for r in g) / 1048576,
             statistics.mean(r[6] for r in g) / 1024, statistics.mean(r[7] for r in g) / 1024))
