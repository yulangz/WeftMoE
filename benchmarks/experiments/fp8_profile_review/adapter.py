from pathlib import Path
import subprocess,os,sys,concurrent.futures,difflib
label=sys.argv[1] if len(sys.argv)>1 else 'detail';base=Path('build/fp8-profile-review');p=Path('src/python/runtime.cpp');old=p.read_text();needle='''    tape->forward = {};
    tape->dispatch = {};''';assert old.count(needle)==1
new=old.replace(needle,'''    // Diagnostic adapter: retain the handle until the BWD trace is downloaded.
#if !WEFTMOE_DEVICE_TIMELINE
    tape->forward = {};
#endif
    tape->dispatch = {};''')
p.write_text(new)
(base/'adapter.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/src/python/runtime.cpp',tofile='b/src/python/runtime.cpp')))
def build(model):
 with (base/f'{label}-{model}-trace-adapter.log').open('w') as log:
  subprocess.run([sys.executable,'tools/build_tensor_runtime.py','--library',str((base/f'{label}-{model}/libweftmoe.so').resolve())],env=dict(os.environ,WEFTMOE_DEVICE_TIMELINE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 print('ADAPTER',model,flush=True)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(build,['qwen35b','qwen122b','flash']))
finally:p.write_text(old)
