#!/usr/bin/env python3
"""Independent protocol audit of JSONL chunk output (no dependency on candidate code).
Usage: python3 tools/audit_coverage.py CORPUS OUTPUT.jsonl [max_bytes]
The supplied benchmark projection contains only included UTF-8 files; ignored or binary
files are reported separately rather than silently hidden by this audit.
Chunk format 3: a line with a `file` key records a file's metadata region (Markdown
front matter), which starts at byte 0; the file's chunks cover the bytes after it.
Files of the original languages' extensions must carry that language; every other file
one language for all its chunks.
"""
import bisect, collections, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); output = pathlib.Path(sys.argv[2]); limit = int(sys.argv[3]) if len(sys.argv)>3 else 4096
chunks = collections.defaultdict(list); ids = set(); maximum = (0,None); counts=collections.Counter(); metadata = {}
for line in output.open():
    c = json.loads(line)
    if 'file' in c:
        span = c['metadata_range']; assert span['start_byte'] == 0, ('metadata_region', c)
        metadata[c['file']] = span['end_byte']; continue
    chunks[c['path']].append(c)
    assert c['id'] not in ids, ('duplicate_id',c); ids.add(c['id'])
    counts[c['language']] += 1
allbytes = 0; empty = []; missing=[]; excluded=[]
for path in sorted(root.rglob('*')):
    if not path.is_file(): continue
    rel=path.relative_to(root).as_posix(); b=path.read_bytes()
    try: text=b.decode('utf8')
    except UnicodeDecodeError: excluded.append(rel); continue
    if b'\0' in b: excluded.append(rel); continue
    cs=sorted(chunks.pop(rel,[]),key=lambda c:c['start_byte']); allbytes+=len(b)
    if not b: empty.append(rel); assert not cs; continue
    meta=metadata.pop(rel,0)
    if not cs and meta!=len(b): missing.append(rel); continue
    starts=[0]+[i+1 for i,v in enumerate(b) if v==10]; previous=meta
    expected={'rs':'rust','py':'python','ts':'typescript','tsx':'typescript','js':'javascript','jsx':'javascript','go':'go','java':'java','c':'c','h':'c','cc':'cpp','cpp':'cpp','hpp':'cpp'}.get(path.suffix.lstrip('.').lower(),cs[0]['language'] if cs else 'text')
    for c in cs:
        a,z=c['start_byte'],c['end_byte']
        assert a==previous and a<z<=len(b), ('coverage',rel,c,previous)
        b[a:z].decode('utf8'); previous=z
        assert c['start_line']==bisect.bisect_right(starts,a), ('start_line',rel,c)
        assert c['end_line']==bisect.bisect_left(starts,z), ('end_line',rel,c)
        assert c['language']==expected, ('language',rel,c)
        assert z-a<=limit, ('source_limit',rel,c)
        rendered=len(rel.encode())+1+z-a
        assert rendered<=24000, ('conservative_token_byte_limit',rel,c)
        if rendered>maximum[0]:maximum=(rendered,rel)
    assert previous==len(b),(rel,previous,len(b))
assert not chunks and not metadata, ('unknown_files',list(chunks)+list(metadata))
assert not missing, ('unaccounted_utf8_files',missing)
print(json.dumps({'utf8_bytes':allbytes,'chunks':len(ids),'languages':counts,'max_rendered_bytes':maximum,'empty_files':len(empty),'declared_binary':len(excluded),'missing_utf8_files':len(missing)},indent=2))
