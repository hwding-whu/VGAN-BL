import context

from src.config import path

if __name__ == '__main__':
    print(path.root)
    print(path.data)
    print(path.plots)
    import src.config
    print(src.config.random.seed)
