"""Isolated Linux PTY probes: compare observable signals to known fixture behavior."""
import errno, fcntl, json, os, pty, re, select, signal, struct, sys, termios, time
from pathlib import Path
PROMPT=re.compile(r'(?:\[[Yy]/[Nn]\]|\([Yy]/[Nn]\))\s*$')
CASES=[
 ('yn_no_newline', 'print("Continue? [Y/n] ",end="",flush=True); print("ANSWER="+input())', True),
 ('stderr_prompt', 'print("Continue? [Y/n] ",end="",file=sys.stderr,flush=True); print("ANSWER="+input())', True),
 ('split_prompt', 'print("Cont",end="",flush=True); time.sleep(.025); print("inue? [Y/n] ",end="",flush=True); print("ANSWER="+input())', True),
 ('ansi_prompt', 'print("\\x1b[32mContinue? [Y/n]\\x1b[0m ",end="",flush=True); print("ANSWER="+input())', True),
 ('raw_key', 'import tty; tty.setraw(0); print("Continue? [Y/n] ",end="",flush=True); print("ANSWER="+os.read(0,1).decode())', True),
 ('dev_tty', 'f=os.open("/dev/tty",os.O_RDWR); os.write(f,b"Continue? [Y/n] "); print("ANSWER="+os.read(f,32).decode().strip())', True),
 ('silent_input', 'print("ANSWER="+input())', True),
 ('unflushed_prompt', 'sys.stdout.write("Continue? [Y/n] "); print("ANSWER="+os.read(0,2).decode())', True),
 ('quiet_sleep', 'time.sleep(.4)', False),
 ('quiet_compute', 'end=time.monotonic()+.4\nwhile time.monotonic()<end: pass', False),
 ('prompt_looking_log', 'print("Continue? [Y/n] ",end="",flush=True); time.sleep(.4)', False),
 ('working_thread_main_reads', 'import threading\ndef work():\n time.sleep(.4); os._exit(0)\nthreading.Thread(target=work).start(); os.read(0,2)', False),
 ('prompt_log_while_working', 'import threading\nprint("Continue? [Y/n] ",end="",flush=True)\ndef work():\n time.sleep(.4); os._exit(0)\nthreading.Thread(target=work).start(); os.read(0,2)', False),
 ('background_reader', 'import threading; threading.Thread(target=lambda:os.read(0,2),daemon=True).start(); time.sleep(.4)', False),
]
ANSI=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

def read(master, duration):
    output=b''; end=time.monotonic()+duration
    while time.monotonic()<end:
        if not select.select([master],[],[],max(0,end-time.monotonic()))[0]: break
        try:
            chunk=os.read(master,65536)
            if not chunk: break
            output+=chunk
        except OSError as e:
            if e.errno==errno.EIO: break
            raise
    return output

rows=[]
for name,code,waiting in CASES:
    pid,master=pty.fork()
    if pid==0:
        os.execv(sys.executable,[sys.executable,'-c','import os,sys,time\nprint("__READY__",flush=True)\n'+code])
    try:
        output=b''; deadline=time.monotonic()+3
        while b'__READY__' not in output and time.monotonic()<deadline: output+=read(master,.02)
        output+=read(master,.1)
        text=output.decode(errors='replace').split('__READY__',1)[-1].lstrip('\r\n')
        try: wchan=Path(f'/proc/{pid}/wchan').read_text()
        except OSError: wchan='unavailable'
        queued=struct.unpack('i',fcntl.ioctl(master,termios.FIONREAD,struct.pack('i',0)))[0]
        writable=bool(select.select([],[master],[],0)[1])
        looks=bool(PROMPT.search(ANSI.sub('',text)))
        response=''
        if waiting:
            os.write(master,b'Y\n'); response=read(master,.15).decode(errors='replace')
            assert 'ANSWER=Y' in response, (name,response)
        rows.append(dict(case=name,application_needs_answer=waiting,prompt_match=looks,write_ready=writable,queued_output_bytes=queued,wchan=wchan,response_ok=('ANSWER=Y' in response if waiting else None)))
    finally:
        try: os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError: pass
        os.waitpid(pid,0); os.close(master)
results={'cases':rows,'prompt_false_positives':[r['case'] for r in rows if r['prompt_match'] and not r['application_needs_answer']], 'prompt_false_negatives':[r['case'] for r in rows if not r['prompt_match'] and r['application_needs_answer']], 'pty_writable_for_every_case':all(r['write_ready'] for r in rows), 'prompt_plus_tty_wait_false_positives':[r['case'] for r in rows if r['prompt_match'] and r['wchan']=='wait_woken' and not r['application_needs_answer']]}
print(json.dumps(results,indent=2))
