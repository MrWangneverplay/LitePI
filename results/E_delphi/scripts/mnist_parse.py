import os, re, glob, statistics, csv

BENCH = '/root/delphi/run/mnist_bench'
OUT = '/mnt/d/王梓杨/项目组/wzy/results/E_delphi/delphi_mnist_raw.csv'
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
        if 'Offline Communication' in line: sec = 'off'; data.setdefault(sec, (0,0))
        elif 'Online Communication' in line: sec = 'on'; data.setdefault(sec, (0,0))
        else:
            c = COMM.search(line)
            if c and sec:
                rd, wr = data[sec]
                if c.group(1) == 'Read': rd = int(c.group(2))
                else: wr = int(c.group(2))
                data[sec] = (rd, wr)
    return off, on, *data.get('off', (0,0)), *data.get('on', (0,0))

rows = []
for p in sorted(glob.glob(os.path.join(BENCH, 'client_L*_r*.log'))):
    m = re.match(r'client_L(\d+)_r(\d+)\.log', os.path.basename(p))
    L, r = int(m.group(1)), int(m.group(2))
    off, on, o_rd, o_wr, n_rd, n_wr = parse_log(p)
    rows.append([L, r, off, on, o_rd, o_wr, n_rd, n_wr])

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['layers','run','offline_s','online_s','off_read','off_write','on_read','on_write'])
    w.writerows(rows)

print('--- Delphi MNIST (official example, 4 threads, loopback) ---')
for L in sorted({r[0] for r in rows}):
    g = [r for r in rows if r[0] == L]
    print('layers=%d n=%d offline=%.3fs±%.3f online=%.3fs±%.3f comm_off=%.1fKB comm_on=%.1fKB'
          % (L, len(g),
             statistics.mean(r[2] for r in g), statistics.pstdev(r[2] for r in g),
             statistics.mean(r[3] for r in g), statistics.pstdev(r[3] for r in g),
             statistics.mean((r[4]+r[5])/1024 for r in g),
             statistics.mean((r[6]+r[7])/1024 for r in g)))
