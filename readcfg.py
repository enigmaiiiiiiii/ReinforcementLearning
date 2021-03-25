import yaml

with open('cfg.yaml') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)
    print(cfg['ENV'])
print(cfg['BATCH_SIZE'])