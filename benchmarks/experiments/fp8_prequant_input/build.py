import concurrent.futures,os,pathlib,subprocess,sys,time,json
root=pathlib.Path.cwd(); base=root/'build/fp8-prequant-input'; label=sys.argv[1]; extra=' '.join(sys.argv[2:]);models=os.environ.get('BUILD_MODELS','qwen35b qwen122b flash').split()
def build(model):
 out=base/f'{label}-{model}';start=time.time()
 env=dict(os.environ,NVCCFLAGS=f'-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I{root}/configs/profiles/{model} {extra}')
 with (base/f'{label}-{model}-build.log').open('w') as log:
  subprocess.run([sys.executable,'build_native.py','lib','--output',str(out/'libweftmoe.so'),'--jobs','2'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (base/f'{label}-{model}-adapter.log').open('w') as log:
  subprocess.run([sys.executable,'tools/build_tensor_runtime.py','--library',str(out/'libweftmoe.so')],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILT',label,model,round(time.time()-start,1),flush=True)
 return dict(model=model,seconds=time.time()-start,flags=env['NVCCFLAGS'])
with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool: results=list(pool.map(build,models))
(base/f'{label}-builds.json').write_text(json.dumps(results,indent=2)+'\n')
