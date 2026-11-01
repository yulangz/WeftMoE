"""Allow the diagnostic-only BWD trace to be downloaded from the live FWD tape."""
from pathlib import Path
path=Path.cwd()/'benchmarks/torchrun_training.py'
source=path.read_text()
old='if args.timeline_dir and (args.backend != "pf" or not args.forward_only or args.verify):'
assert source.count(old)==1
source=source.replace(old,'if args.timeline_dir and (args.backend != "pf" or args.verify):')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
