import yaml

with open('cfg.yaml') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)
    print(cfg['ENV'])
print(cfg['BATCH_SIZE'])

#%%
with open('cfg.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

ENV = cfg['ENV']
GAMMA = cfg['GAMMA']
MAX_STEPS = cfg['MAX_STEPS']
NUM_EPISODES = cfg['NUM_EPISODES']
CAPACITY = cfg['CAPACITY']
BATCH_SIZE = cfg['BATCH_SIZE']