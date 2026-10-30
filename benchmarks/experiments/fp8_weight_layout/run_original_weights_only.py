"""Experiment-only adapter: ask native RS build to train with original weights only."""
import os,runpy,sys
import weftmoe.fp8 as fp8
original_prepare=fp8.prepare_fp8_weights
original_wait=fp8.PreparedFP8Weights.wait
path=sys.argv.pop(1)
def prepare(w1,w2,**kwargs):
    # The parity test keeps a separate full-layout global reference. Only the
    # actual per-rank weights omit transposes; the benchmark has only local weights.
    reference = 'torchrun_fp8' in path and w1.shape[0] == int(os.environ.get('TEST_EXPERTS','32'))
    if not reference:kwargs['training']=False
    result=original_prepare(w1,w2,**kwargs)
    if not reference:
        assert result.w1_t is None and result.w2_t is None
        size=sum(t.numel()*t.element_size() for t in result._tensors())
        print('FP8_RS_ORIGINAL_WEIGHTS_ONLY',size,flush=True)
    return result
def wait(self,*,training=False):
    original_wait(self,training=False)
fp8.prepare_fp8_weights=prepare
fp8.PreparedFP8Weights.wait=wait
sys.argv[0]=path
runpy.run_path(path,run_name='__main__')
