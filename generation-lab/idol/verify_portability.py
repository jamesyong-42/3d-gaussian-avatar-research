"""Run scoped regressions and save their actual output in a selected evidence folder."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / 'generation-lab'
sys.path.insert(0, str(ROOT / 'web/backend'))
from settings import DATA


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r'(portability|deformation)-\d{8}-\d{6}-[0-9a-f]{6}', sys.argv[1]):
        raise ValueError('Supply one portability or deformation run ID')
    out = LAB / 'reports/idol' / sys.argv[1]
    if not out.is_dir(): raise ValueError('Run does not exist')
    node = shutil.which('node')
    if not node: raise RuntimeError('Node is required')
    baseline = DATA / 'avatars/example'
    def baseline_hashes():
        return {name: hashlib.sha256((baseline/name).read_bytes()).hexdigest() for name in ('avatar.json', 'avatar.bin')}
    expected = baseline_hashes() if (baseline/'avatar.json').is_file() and (baseline/'avatar.bin').is_file() else None
    checks = [
        ([sys.executable, '-m', 'unittest', 'discover', '-s', 'generation-lab/idol', '-p', 'test_*.py', '-v'], ROOT),
        ([sys.executable, 'generation-lab/test_generation.py'], ROOT),
        ([sys.executable, 'generation-lab/test_packaging.py'], ROOT),
        ([node, '--test', '--test-isolation=none', 'generation-lab/idol/*.test.mjs'], ROOT),
        ([node, '--test', '--test-isolation=none', 'tests/*.test.mjs'], ROOT / 'web'),
        ([node, 'node_modules/typescript/bin/tsc', '--noEmit'], ROOT / 'web')]
    evidence = dict(date=datetime.now(timezone.utc).isoformat(), scope='Isolated IDOL lab plus unchanged main-app unit/type regressions', checks=[],
                    mainApplicationBrowserSuiteRerun=False, mainApplicationChanged=False, browserAvatarPromoted=False)
    for command,cwd in checks:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), timeout=180)
        evidence['checks'].append(dict(command=command,cwd=str(cwd),exitCode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        print(json.dumps(dict(command=command,exitCode=result.returncode)),flush=True)
    actual = baseline_hashes() if expected is not None else None
    evidence.update(baselineSha256=actual,baselineChecked=expected is not None,baselineUnchanged=actual==expected if expected is not None else None)
    result = subprocess.run(['git','status','--porcelain'],cwd=LAB/'vendors/IDOL',capture_output=True,text=True,
                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=30)
    evidence['vendorCheckoutClean'] = result.returncode==0 and not result.stdout.strip()
    evidence['pass'] = all(c['exitCode']==0 for c in evidence['checks']) and evidence['baselineUnchanged'] is not False and evidence['vendorCheckoutClean']
    (out/'regression-checks.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({k:evidence[k] for k in ['pass','baselineUnchanged','vendorCheckoutClean']}))
    return 0 if evidence['pass'] else 1


if __name__ == '__main__': raise SystemExit(main())
