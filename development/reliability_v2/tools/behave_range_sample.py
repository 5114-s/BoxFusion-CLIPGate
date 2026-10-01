"""Read selected members of official BEHAVE ZIPs without downloading archives."""
import argparse
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/behave_motion_probe_20260914'
BASE = 'https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/'


class RemoteFile(io.RawIOBase):
    def __init__(self, url, budget=32*1024*1024, metadata=None):
        self.url, self.pos, self.used, self.budget = url, 0, 0, budget
        if metadata is not None:
            self.size, self.etag = metadata['archive_bytes'], metadata['etag']
            return
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers.get('ETag')

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos+n)
        if end <= self.pos:
            return b''
        if self.used + end-self.pos > self.budget:
            raise RuntimeError('Explicit download budget exceeded')
        headers = {'Range': f'bytes={self.pos}-{end-1}', 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Range'] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=45) as r:
            assert r.status == 206, 'Server ignored Range; refusing whole archive'
            assert r.headers['Content-Range'] == f'bytes {self.pos}-{end-1}/{self.size}'
            data = r.read(end-self.pos+1)
        assert len(data) == end-self.pos
        self.used += len(data)
        self.pos = end
        return data


def index(name):
    remote = RemoteFile(BASE + name)
    with zipfile.ZipFile(remote) as z:
        members = [{'name': i.filename, 'offset': i.header_offset, 'compressed': i.compress_size,
                    'size': i.file_size, 'method': i.compress_type, 'crc': i.CRC}
                   for i in z.infolist() if not i.is_dir()]
    record = {'url': remote.url, 'archive_bytes': remote.size, 'etag': remote.etag,
              'index_download_bytes': remote.used, 'members': members}
    (OUT / (name + '.index.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(name, 'members', len(members), 'index_bytes', remote.used, flush=True)
    return record


def member(record, entry, dest):
    target = dest / entry['name']
    assert dest.resolve() in target.resolve().parents
    if target.exists():
        data = target.read_bytes()
        assert len(data) == entry['size'] and zlib.crc32(data) == entry['crc']
        return {'path': str(target), 'cached': True, 'network_bytes': 0, 'sha256': hashlib.sha256(data).hexdigest()}
    remote = RemoteFile(record['url'], budget=64*1024*1024, metadata=record)
    assert remote.etag == record['etag'] and remote.size == record['archive_bytes']
    remote.seek(entry['offset'])
    block = remote.read(entry['compressed']+1024)
    assert block[:4] == b'PK\x03\x04'
    name_len, extra_len = struct.unpack_from('<HH', block, 26)
    start = 30+name_len+extra_len
    assert start <= 1024
    packed = block[start:start+entry['compressed']]
    data = zlib.decompress(packed, -15) if entry['method'] == 8 else packed
    assert len(data) == entry['size'] and zlib.crc32(data) == entry['crc']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {'path': str(target), 'cached': False, 'network_bytes': remote.used, 'sha256': hashlib.sha256(data).hexdigest()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', nargs='+', default=[])
    p.add_argument('--fetch-plan', type=Path)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.index:
        record = index(name)
        print('roots', sorted({x['name'].split('/')[0] for x in record['members']}), flush=True)
    if args.fetch_plan:
        plan = json.loads(args.fetch_plan.read_text())
        wanted = set(plan['files'])
        jobs = []
        for f in OUT.glob('*.zip.index.json'):
            record = json.loads(f.read_text())
            jobs.extend((record, e) for e in record['members'] if e['name'] in wanted)
        assert len(jobs) == len(wanted)
        assert sum(e['compressed'] for _, e in jobs) <= plan.get('max_compressed_bytes', 40*1024*1024)
        receipts = []
        dest = ROOT / 'data_behave_probe'
        with ThreadPoolExecutor(4) as pool:
            for r in pool.map(lambda job: member(job[0], job[1], dest), jobs):
                receipts.append(r)
                print(Path(r['path']).name, r['network_bytes'], flush=True)
                args.fetch_plan.with_suffix('.receipts.json').write_text(json.dumps(receipts, indent=2)+'\n')
