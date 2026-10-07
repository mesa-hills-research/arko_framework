#!/usr/bin/env python3
"""Offline subprocess CLI integration + kill/restart/termination harness."""
import json,os,pathlib,select,signal,subprocess,sys,tempfile,time
bin=str(pathlib.Path(sys.argv[1]).resolve())
def run(*args):return subprocess.run([bin,*map(str,args)],check=True,capture_output=True,text=True)
def read_event(p):
    assert select.select([p.stdout],[],[],5)[0], 'watch timed out'
    s=p.stdout.readline();assert s, p.stderr.read();return json.loads(s)
def start(r,s):return subprocess.Popen([bin,'watch',str(r),str(s)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
with tempfile.TemporaryDirectory() as t:
    r=pathlib.Path(t)/'root';r.mkdir();s=r/'state';
    (r/'.gitignore').write_text('ignored/\n*.bin\n');(r/'ignored').mkdir();(r/'ignored'/'bad.rs').write_text('fn forbidden(){}')
    (r/'.hidden').write_text('hidden utf8 é🙂\n');(r/'path.rs').write_text('fn durable_replay() {}\nfn alpha() {}\n')
    for n in range(20):(r/f'{n}.rs').write_text(f'fn alpha_{n}(){{}}\n')
    for _ in range(2):run('index',r,s)
    chunks=[json.loads(s) for s in run('chunk',r).stdout.splitlines()]
    assert '.hidden' in {c['path'] for c in chunks};assert not any('state/' in c['path'] for c in chunks)
    # Gitignored files are indexed by default; --respect-ignore honours .gitignore.
    assert 'ignored/bad.rs' in {c['path'] for c in chunks}
    honoured=[json.loads(s) for s in run('chunk','--respect-ignore',r).stdout.splitlines()]
    assert not any('ignored/' in c['path'] for c in honoured) and len(honoured)==len(chunks)-1
    hits=[json.loads(s) for s in run('search',s,'alpha').stdout.splitlines()];assert len(hits)==10
    one=[json.loads(s) for s in run('search',s,'durable_replay').stdout.splitlines()];assert one[0]['path']=='path.rs'
    p=start(r,s);d=read_event(p);seq=d['sequence'];assert len(d['added'])==len(chunks)
    # SIGKILL after initial delivery; ack may or may not have committed. Either is legal.
    p.kill();p.wait(timeout=3)
    (r/'path.rs').write_text('fn inserted() {}\nfn durable_replay() {}\nfn alpha() {}\n')
    p=start(r,s);d=read_event(p)
    if d['sequence']==seq:d=read_event(p)
    assert d['sequence']>seq and len(d['changed'])==2
    # Burst creates, rename saves, deletes, then clean SIGTERM and an offline edit.
    for n in range(30):
        tmp=r/'save.tmp';tmp.write_text(f'fn item_{n}(){{}}\n');tmp.replace(r/f'new{n}.rs')
    for n in range(10):(r/f'{n}.rs').unlink()
    d=read_event(p);assert d['sequence']>seq
    p.send_signal(signal.SIGTERM);p.wait(timeout=5);assert p.returncode==0,p.stderr.read()
    state=json.loads((s/'watch.json').read_text());assert len(state['chunks'])==len([json.loads(x) for x in run('chunk',r).stdout.splitlines()])
    seq=state['sequence'];(r/'offline.py').write_text('def offline():\n    return 1\n')
    p=start(r,s);d=read_event(p)
    while d['sequence']<=seq:d=read_event(p)
    assert any(c['path']=='offline.py' for c in d['added'])
    p.send_signal(signal.SIGTERM);p.wait(timeout=5);assert p.returncode==0
    # Broken output pipe must persist unacked delta and shut down with a diagnostic.
    p=start(r,s);p.stdout.close();(r/'broken.py').write_text('print("broken pipe")\n')
    p.wait(timeout=5);assert p.returncode!=0
    state=json.loads((s/'watch.json').read_text());assert state['pending'] and state['pending'][-1]['sequence']>state['acked']
print('CLI PASS: repeat index/reload, hidden/ignore/state paths, <=10 best-first search, SIGKILL restart, burst edits, SIGTERM, broken-pipe unacked persistence')
