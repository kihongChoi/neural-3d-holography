"""Finish the sealed 5 mm test at the user's verified manually placed stage.

No stage motion or homing commands are issued. Fresh calibration is at 5 mm.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import traceback
import torch  # Before vendor DLLs.

from .integrated_rgb_collect import Collector, write_json
from .dpac_lut_session import now, sha
from .dpac_preflight import run as preflight
from . import geometry_registration as geo
from . import step01_dot_grid as grid
from .hardware import Bench
from .readybeam import ReadyBeam, exclusive
from .stage import Stage


class StationaryStage:
    def __init__(self, stage, target):
        self.stage=stage
        self.target=target
        self.moves=[]

    def status(self):return self.stage.status()
    require_stationary=staticmethod(Stage.require_stationary)

    def move(self,target):
        state=self.status()
        self.require_stationary(state)
        if abs(target-self.target)>.000001 or abs(state['position_mm']-self.target)>.002:
            raise RuntimeError(f'Stationary-only target mismatch: {target}, {state}')
        return state


def run(root):
    root=Path(root).resolve()
    control=root/'stationary_5mm_resume'
    control.mkdir(exist_ok=False)
    shutil.copy2(__file__,control/'runner.py')
    shutil.copy2('docs/rgb-asm-nh3d-lut-plan.md',control/'plan_at_resume.md')
    pre=preflight(control)
    assert all(x['ok'] for x in pre['checks'].values())
    assert pre['checks']['display']['result']['expected_present']
    state=pre['checks']['stage']['result']['state']
    Stage.require_stationary(state)
    assert abs(state['position_mm']-95.780)<.002
    c=Collector(root)
    c.report.update(stationary_only=True,runner_sha256=sha(__file__),
                    preflight_path=str(control.relative_to(root)/'preflight.json'),
                    protocol='Fresh 5 mm calibration/routing, no zero-plane visit or stage motion')
    remaining=[r for r in c.plan['records'] if r['id'] not in c.done]
    assert len(c.done)==8100 and len(remaining)==100
    assert all(r['split']=='test' and r['design_mm']==5. for r in remaining)
    try:
        frames=geo.cached_patterns(5.,root/'calibration_patterns')
        c.generator.load(remaining[0])
        with exclusive():
            c.laser=ReadyBeam()
            try:
                c.report['standby']=c.laser.configure('Digital')
                with Stage(monitor_only=True) as real_stage:
                    c.stage=StationaryStage(real_stage,95.780)
                    c.report['stage_start']=c.stage.move(95.780)
                    try:
                        with Bench(dict(grid.C,settle_s=.25)) as c.b:
                            if (c.b.camera.Width.GetValue(),c.b.camera.Height.GetValue(),c.b.camera.OffsetX.GetValue(),c.b.camera.OffsetY.GetValue())!=(4096,3000,0,0):
                                raise RuntimeError('Full sensor geometry mismatch')
                            c.report['camera_original']=c.b.original
                            cal=c.calibrate(5.,frames,route=False)
                            c.routing(cal)
                            c.collect_depth(5.,cal)
                    finally:
                        # Stage readback is independent of laser UI shutdown success.
                        try:
                            c.report['stage_end']=c.stage.status()
                            c.report['moves']=[]
                        except Exception as exc:c.report['stage_end_error']=repr(exc)
            finally:
                try:
                    c.report['final_standby']=c.laser.configure('Digital')
                except Exception as exc:
                    c.report['shutdown_first_error']=repr(exc)
                    c.report['final_standby']=ReadyBeam().configure('Digital')
        counts=Counter(r['split'] for r in c.done.values())
        assert dict(counts)==c.plan['expected_counts']
        Stage.require_stationary(c.report['stage_end'])
        assert abs(c.report['stage_end']['position_mm']-95.780)<.002
        c.report['state']='complete'
        write_json(root/'dataset_raw.json',dict(version='integrated-rgb-raw-v1',complete=True,
                   source_type='physical_camera',completed_kst=now(),counts=dict(counts),
                   plan_sha256=c.plan_hash,preprocessing='deferred',
                   records=[c.done[i] for i in sorted(c.done)]))
        print('COMPLETE',dict(counts),flush=True)
    except BaseException as exc:
        c.report.update(state='interrupted',error=repr(exc),traceback=traceback.format_exc())
        print('INTERRUPTED',repr(exc),flush=True)
        raise
    finally:
        c.save();c.journal.close();c.scene_journal.close()
        with Path('docs/rgb-asm-nh3d-lut-plan.md').open('a',encoding='utf8') as f:
            f.write(f'\n\n- {now()} stationary 5 mm `{root.name}/{c.out.name}`: {c.report["state"]}, 완료 {len(c.done)}/8200. 근거 `{c.out.relative_to(Path.cwd())}/status.json`, `stationary_5mm_resume/`, `completed_blocks/`. 오류 {c.report.get("error","없음")}. Stage {c.report.get("stage_end")}, final standby {c.report.get("final_standby")}. 자동이동/호밍없음, 후처리/훈련미실행.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    run(p.parse_args().root)
