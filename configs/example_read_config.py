import yaml

data=None
with open('coco.yaml', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    print(data)

print(type(data['voc']['clip']))