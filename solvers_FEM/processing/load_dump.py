import yaml

def load_yml(path):
    with open(path) as file:
        dict = yaml.load(file, Loader=yaml.Loader)
    return dict

def dump_yml(dump,path):
    with open(path,'w') as file:
        file.write(yaml.dump(dump))

def load_txt(path):
    with open(path) as file:
        txt = file.read()
    return txt

def dump_txt(dump,path):
    with open(path,'w') as file:
        file.write(dump)