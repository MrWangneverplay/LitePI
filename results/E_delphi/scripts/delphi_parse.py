#!/usr/bin/env python3
"""Parse Delphi official latency/comm logs into CSV (E_delphi baseline)."""
import os, re, csv, glob, statistics, sys

BENCH = '/root/delphi/run/bench'
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/王梓杨/项目组/wzy/results/E_delphi'
THREADS = os.environ.get('THREADS', '4')
HW = os.environ.get('HW', 'Intel Core Ultra 5 125H / WSL2 / 18 logical cores')
os.makedirs(OUTDIR, exist_ok=True)

DUR = re.compile(r'^(?:.*?)End:\s+Client (offline|online) phase\.+([0-9.]+)\s*(ms|s|µs|us|m)$')
COMM = re.compile(r'(Read|Wrote) ([0-9]+) bytes')

def parse_dur(d, unit):
    d = float(d)
    if unit == 'ms':
        return d / 1000.0
    if unit in ('µs', 'us'):
        return d / 1e6
    if unit == 'm':
        return d * 60.0
    return d

def parse_log(path):
    off = on = None
    sections = {}
    cur = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = DUR.match(line)
            if m:
                val = parse_dur(m.group(2), m.group(3))
                if m.group(1) == 'offline':
                    off = val
                else:
                    on = val
            if 'Offline Communication' in line:
                cur = 'off'
                sections.setdefault('off', [])
            elif 'Online Communication' in line:
                cur = 'on'
                sections.setdefault('on', [])
            else:
                m = COMM.search(line)
                if m and cur:
                    sections[cur].append((m.group(1), int(m.group(2))))
    def pair(sec):
        rd = wr = None
        for kind, val in sections.get(sec, []):
            if kind == 'Read':
                rd = val
            else:
                wr = val
        return rd, wr
    off_rd, off_wr = pair('off')
    on_rd, on_wr = pair('on')
    return off, on, off_rd, off_wr, on_rd, on_wr

def parse_time(path):
    wall = rss = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if 'Elapsed (wall clock)' in line:
                t = line.split('): ')[1].strip()
                parts = [float(x) for x in t.split(':')]
                wall = parts[-1] + (parts[-2] * 60 if len(parts) >= 2 else 0) + (parts[-3] * 3600 if len(parts) == 3 else 0)
            if 'Maximum resident set size' in line:
                rss = int(line.split(':')[1].strip())
    return wall, rss

rows = []
for clog in sorted(glob.glob(os.path.join(BENCH, 'client_L*_r*.log'))):
    m = re.match(r'client_L(\d+)_r(\d+)\.log', os.path.basename(clog))
    if not m:
        continue
    L, r = int(m.group(1)), int(m.group(2))
    off, on, off_rd, off_wr, on_rd, on_wr = parse_log(clog)
    cw, crss = parse_time(clog[:-4] + '.time')
    spath = os.path.join(BENCH, f'server_L{L}_r{r}.log')
    sw, srss = parse_time(spath[:-4] + '.time')
    rows.append(dict(layers=L, run=r, threads=THREADS, hw=HW,
                     offline_s=off, online_s=on,
                     client_off_read=off_rd, client_off_write=off_wr,
                     client_on_read=on_rd, client_on_write=on_wr,
                     client_rss_kb=crss, server_rss_kb=srss,
                     client_wall_s=cw, server_wall_s=sw))

rows.sort(key=lambda x: (x['layers'], x['run']))
fields = list(rows[0].keys())
with open(os.path.join(OUTDIR, 'delphi_minionn_raw.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# summary mean/std
with open(os.path.join(OUTDIR, 'delphi_minionn_summary.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layers', 'n', 'offline_s_mean', 'offline_s_std', 'online_s_mean', 'online_s_std',
                'client_comm_off_KB', 'client_comm_on_KB', 'client_rss_MB', 'server_rss_MB'])
    for L in sorted({r['layers'] for r in rows}):
        g = [r for r in rows if r['layers'] == L]
        w.writerow([L, len(g),
                    round(statistics.mean(r['offline_s'] for r in g), 3),
                    round(statistics.pstdev(r['offline_s'] for r in g), 3),
                    round(statistics.mean(r['online_s'] for r in g), 3),
                    round(statistics.pstdev(r['online_s'] for r in g), 3),
                    round(statistics.mean((r['client_off_read'] + r['client_off_write']) / 1024.0 for r in g), 1),
                    round(statistics.mean((r['client_on_read'] + r['client_on_write']) / 1024.0 for r in g), 1),
                    round(statistics.mean(r['client_rss_kb'] for r in g) / 1024.0, 1),
                    round(statistics.mean(r['server_rss_kb'] for r in g) / 1024.0, 1)])

print('rows:', len(rows))
for r in rows:
    print(r)
