"""Continue the study acquisition after the first measured production block."""
import argparse
import json
from pathlib import Path
import time

from .rgb_study_block import acquire, publish_status
from .rgb_study_manifest import VERSION, DEPTHS, TRAIN_DEPTHS, PATTERNS


def completed(folder,z,first,last):
    report=json.loads((folder/'status.json').read_text(encoding='utf8'))
    expected={(i,ch) for i in range(first,last) for ch in 'RGB'}
    accepted={(r['id'],r['channel']) for r in report['records'] if r['accepted']}
    if (report.get('error') or report['plan']!=VERSION or report['z']!=z or len(report['records'])!=len(expected)
            or accepted!=expected or report.get('final_standby',{}).get('mode')!='Digital'):
        raise ValueError(f'Incomplete block: {folder}; inspect before continuing')
    return len(accepted)


def run(args):
    root=args.out
    root.mkdir(parents=True,exist_ok=True)
    profile=dict(plan=VERSION,added_nd_od=.4,patterns=str(args.patterns),cal=str(args.calibration),
                 routing=str(args.routing),signs=[str(p) for p in args.sign_runs],first_block=str(args.first_block))
    config=root/'config.json'
    if config.exists():
        if json.loads(config.read_text())!=profile: raise ValueError('Campaign inputs changed')
    else:
        with config.open('x') as f: json.dump(profile,f,indent=2)
    done=completed(args.first_block,DEPTHS[0],0,32)
    report=dict(accepted=done,planned=PATTERNS*len(DEPTHS)*3,completed_blocks=[str(args.first_block)])
    def update(**fields):
        report.update(fields)
        publish_status(root,report)
    update(state='running')
    try:
        starts=[0,*range(32,PATTERNS,128),PATTERNS]
        for n,(first,last) in enumerate(zip(starts,starts[1:])):
            depths=DEPTHS
            for z in depths[n%len(depths):]+depths[:n%len(depths)]:
                if first==0 and z==depths[0]: continue
                folder=root/f'{first:04}_{last:04}_z{z:g}'
                if (root/'STOP_REQUESTED').exists():
                    update(state='stopped_between_blocks')
                    return
                if not folder.exists():
                    update(state='waiting_for_codes',current_block=str(folder))
                    deadline=time.monotonic()+3600
                    while not (args.patterns/f'{last-1:04}.json').is_file():
                        if time.monotonic()>deadline: raise TimeoutError('Pattern generator stalled')
                        time.sleep(10)
                    update(state='acquiring')
                    acquire(args.patterns,folder,z,first,last,args.calibration,args.routing,False,args.sign_runs)
                done+=completed(folder,z,first,last)
                report['completed_blocks'].append(str(folder))
                update(accepted=done)
        if done!=report['planned']: raise RuntimeError(f'Incomplete dataset ({done})')
        update(state='complete')
    except BaseException as e:
        update(state='interrupted',error=repr(e))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('patterns',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--routing',required=True,type=Path)
    p.add_argument('--sign-runs',required=True,type=Path,nargs=len(TRAIN_DEPTHS))
    p.add_argument('--first-block',required=True,type=Path)
    run(p.parse_args())
