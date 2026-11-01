import os,sys,shutil,pathlib
command=[sys.executable,str(pathlib.Path.cwd()/'benchmarks/torchrun_training.py'),*sys.argv[1:]]
if int(os.environ['LOCAL_RANK'])==2:
 command=[shutil.which('ncu'),'--target-processes','application-only','--replay-mode','kernel','--profile-from-start','off','--clock-control','none','--cache-control','none','--kernel-name-base','demangled','--kernel-name',os.environ['WEFTMOE_NCU_KERNEL'],'--launch-count','1','--metrics',os.environ['WEFTMOE_NCU_METRICS'],'--export',os.environ['WEFTMOE_NCU_OUTPUT'],'--force-overwrite',*command]
os.execv(command[0],command)
