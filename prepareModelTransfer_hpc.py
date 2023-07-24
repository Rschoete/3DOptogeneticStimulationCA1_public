import glob
import os
import shutil

model_Dir = './Model'
dir = './Model_HPC'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

filesin_model = [x for x in glob.glob(model_Dir+'/*')  if os.path.isfile(x)]
for f in filesin_model:
    shutil.copy2(f, dir)
for x in ['Mods', 'morphologies']:
    shutil.copytree(os.path.join(model_Dir,x),os.path.join(dir,x))

miglioreModels = os.listdir(os.path.join(model_Dir,'MiglioreModels'))
for x in miglioreModels:
    target = os.path.join(dir,"MiglioreModels",x)
    source = os.path.join(model_Dir,"MiglioreModels",x)
    os.makedirs(target)
    for y in ["mechanisms", "morphology"]:
        shutil.copytree(os.path.join(source,y),os.path.join(target,y))
    os.makedirs(os.path.join(target,'checkpoints'))
    source_hocs = glob.glob(os.path.join(source,"checkpoints/*.hoc"))
    for sh in source_hocs:
        shutil.copy2(sh,os.path.join(target,'checkpoints'))
#shutil.copytree('bar', dir)  # Will fail if `foo` exists
#shutil.copytree('baz', 'foo', dirs_exist_ok=True)