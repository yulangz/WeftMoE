import os,sys,pathlib,subprocess,json,re,time
root=pathlib.Path.cwd();base=root/'build/fp8-profile-review';out=base/'ncu';out.mkdir(exist_ok=True)
groups={'basic':'gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active','stalls':'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct'}
records=[]
for group,metrics in groups.items():
 failed=False
 for mode in ['fwd','bwd']:
  for dtype in ['fp8','bf16']:
   name=f'{group}-{mode}-{dtype}';stem=out/name
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NCCL_DEBUG='WARN',TEST_FP8_COMPACT='1',WEFTMOE_TENSOR_EXTENSION=str(next((base/'baseline-qwen35b').glob('_runtime*.so'))),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64',WEFTMOE_NCU_OUTPUT=str(stem),WEFTMOE_NCU_METRICS=metrics,WEFTMOE_NCU_KERNEL='regex:.*weftmoe_pf::'+('local::kernel' if mode=='fwd' else 'backward_stream::kernel')+'.*')
   env.pop('WEFTMOE_AUTOTUNE_TABLE',None)
   command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(base/'ncu_rank.py'),'--backend','pf','--model','qwen35b','--tokens','20480','--token-std-fraction','0','--warmups','3','--samples','1','--stream-ordered','--ncu-range','--forward-only' if mode=='fwd' else '--backward-only']
   if dtype=='fp8':command+=['--fp8','--weight-grad-dtype','bf16']
   print('NCU',name,flush=True)
   if stem.with_suffix('.ncu-rep').exists():
    p=subprocess.CompletedProcess(command,0)
   else:
    with stem.with_suffix('.log').open('w') as log:p=subprocess.run(['timeout','--kill-after=10s','90s',*command],env=env,stdout=log,stderr=subprocess.STDOUT)
   log=stem.with_suffix('.log').read_text();passes=re.findall(r'(\d+) pass(?:es)?',log)
   records.append(dict(name=name,returncode=p.returncode,passes=passes,command=command,metrics=metrics))
   (out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
   if p.returncode or passes!=['1']:
    print('INVALID',name,p.returncode,passes,flush=True);failed=True;break
   with stem.with_suffix('.csv').open('w') as csv:
    subprocess.run(['ncu','--import',str(stem.with_suffix('.ncu-rep')),'--csv','--page','raw','--print-units','base'],stdout=csv,stderr=subprocess.STDOUT,check=True)
   print('DONE',name,'one pass',flush=True)
  if failed:break
