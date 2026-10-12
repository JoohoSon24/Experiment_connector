from pathlib import Path
import sys,os,json,traceback
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'experiments/recipe';OUT.mkdir(parents=True,exist_ok=True)
log=(OUT/'console.log').open('a',buffering=1);sys.stdout=log;sys.stderr=log
code=1
try:
 from run_recipe_queue import main
 main();code=0
except BaseException:traceback.print_exc()
finally:(OUT/'exit_code.json').write_text(json.dumps({'exit_code':code}));log.flush()
sys.exit(code)
