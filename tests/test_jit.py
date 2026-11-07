"""Cache precedence, shape/bucket matching, build failure and concurrent builders."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sysconfig
from types import SimpleNamespace

import pytest

from weftmoe import jit, tuning


def key(**updates):
    return dict(gpu='test SM90', interconnect='nvlink', dtype='bfloat16', hidden=2048,
                intermediate=512, experts=256, top_k=8, nodes=1, gpus_per_node=8,
                input_layout='dp', mode='training') | updates


def entry(interval=(0,8192), **updates):
    header = jit.default_header(2048,512)
    return dict(key=key(), source='test-source', tokens=list(interval), header=header,
                header_digest=tuning.digest(header), batch_invariant=True,
                window_tokens=2048, contribution_windows=2) | updates


def find(path, tokens=8192, **kwargs):
    return tuning.find_entry(path,key(),tokens,source='test-source',
                             canonical_header=jit.default_header(2048,512),**kwargs)


def test_exact_shape_and_inclusive_buckets(tmp_path):
    path=tmp_path/'tuning.json'
    a,b=entry(),entry((8193,20480),window_tokens=1024)
    tuning.publish_entry(path,a); tuning.publish_entry(path,b)
    assert find(path,0)==a and find(path,8192)==a
    assert find(path,8193)==b and find(path,20480)==b
    assert find(path,20481) is None
    for change in [dict(hidden=3072),dict(experts=128),dict(top_k=4),dict(dtype='fp8'),dict(mode='decode')]:
        assert tuning.find_entry(path,key(**change),8192,source='test-source',canonical_header=a['header']) is None
    with pytest.raises(ValueError,match='overlapping'):
        tuning.publish_entry(path,entry((4096,16384)))


def test_stale_and_noninvariant_configs_do_not_override_defaults(tmp_path):
    path=tmp_path/'tuning.json'
    a=entry(batch_invariant=False)
    tuning.publish_entry(path,a)
    assert find(path) is None
    assert find(path,batch_invariant=False)==a
    a=entry(source='old-source')
    tuning.publish_entry(path,a)
    assert find(path) is None
    a=entry(); a['header']=a['header'].replace('WEFTMOE_FWD_TILE_K 64','WEFTMOE_FWD_TILE_K 128')
    a['header_digest']=tuning.digest(a['header']); tuning.publish_entry(path,a)
    assert find(path) is None
    assert find(path,batch_invariant=False)==a


def test_table_corruption_is_not_silently_accepted(tmp_path):
    path=tmp_path/'tuning.json'; a=entry(header_digest='bad')
    path.write_text(json.dumps(dict(schema=2,entries=[a])))
    with pytest.raises(ValueError,match='digest'):
        find(path)
    a=entry(); path.write_text(json.dumps(dict(schema=2,entries=[a,a])))
    with pytest.raises(ValueError,match='overlapping'):
        find(path)


def fake_artifact(directory):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/('_runtime'+sysconfig.get_config_var('EXT_SUFFIX'))
    path.write_bytes(b'fake-library')
    (directory/'ready.json').write_text(json.dumps(dict(files={path.name:hashlib.sha256(path.read_bytes()).hexdigest()})))
    return path


def test_request_cache_precedes_tuning_and_survives_missing_table(tmp_path,monkeypatch):
    monkeypatch.setenv('WEFTMOE_ROOT',str(tmp_path))
    monkeypatch.delenv('WEFTMOE_AUTOTUNE_TABLE',raising=False)
    monkeypatch.setattr(jit,'build_context',lambda:dict(source='test-source',arch=(9,0)))
    built=[]
    def build(header,context):
        built.append(header)
        return fake_artifact(tmp_path/tuning.digest(header))
    monkeypatch.setattr(jit,'compile_artifact',build)
    first=jit.prepare(key(),4096)
    assert first['origin']=='default'
    tuning.publish_entry(tuning.default_table(),entry())
    assert jit.prepare(key(),4096)==first and len(built)==1
    second=jit.prepare(key(),8192)
    assert second['origin']=='tuned' and second['tuning']['window_tokens']==2048
    tuning.default_table().unlink()
    assert jit.prepare(key(),8192)==second and len(built)==2
    Path(second['extension']).write_bytes(b'corrupted')
    assert jit.prepare(key(),8192)['origin']=='default' and len(built)==3


def test_concurrent_compile_publishes_once_and_retries_failure(tmp_path,monkeypatch):
    monkeypatch.setenv('WEFTMOE_ROOT',str(tmp_path))
    monkeypatch.delenv('NVCCFLAGS',raising=False)
    calls=[]
    fail=[True]
    def run(command,**kwargs):
        calls.append(command)
        if fail[0]:
            return SimpleNamespace(returncode=1)
        if '--output' in command:
            Path(command[command.index('--output')+1]).write_bytes(b'native')
        else:
            path=Path(command[command.index('--library')+1]).parent
            (path/('_runtime'+sysconfig.get_config_var('EXT_SUFFIX'))).write_bytes(b'binding')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(jit.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='compilation failed'):
        jit.compile_artifact('#pragma once\n',{})
    assert not list(tmp_path.rglob('ready.json'))
    fail[0]=False
    with ThreadPoolExecutor(4) as pool:
        paths=list(pool.map(lambda _:jit.compile_artifact('#pragma once\n',{}),range(4)))
    assert len(set(paths))==1 and len(calls)==3
    assert jit._ready(paths[0].parent)


def test_default_shape_is_independent_of_model_name():
    for h,i in [(2048,512),(3072,1024),(4096,2048),(2560,768)]:
        header=jit.default_header(h,i)
        values=tuning.macros(header)
        assert values['WEFTMOE_FWD_MAIN_HIDDEN']==str(h)
        assert values['WEFTMOE_BWD_MAIN_INTERMEDIATE']==str(i)
    with pytest.raises(ValueError):
        jit.default_header(2049,512)
    with pytest.raises(ValueError, match='>= 512'):
        jit.default_header(256,256)
