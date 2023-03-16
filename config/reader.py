from json import load
# чтение файла с конфигураций проекта
def read_config(path):
    cfg = open(path, 'r')
    return load(cfg)