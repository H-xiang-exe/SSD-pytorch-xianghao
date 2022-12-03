class Container(object):
    def __init__(self, *args, **kwargs):
        self._data_dict = dict(*args, **kwargs)

    def __setattr__(self, key, value):
        """在类实例的每个属性进行赋值时，都会首先调用__setattr__()方法，并在
        __setattr__()方法中将属性名和属性值添加到类实例的__dict__属性中"""
        object.__setattr__(self, key, value)

    def __getitem__(self, item):
        """类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值"""
        return self._data_dict[item]

    def __iter__(self):
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def __repr__(self):
        """通常情况下，直接输出某个实例化对象，本意往往是想了解该对象的基本信息，例如该对象有哪些属性，它们的值各是多少等等。
        但默认情况下，得到的信息只会是“类名+object at+内存地址”，对我们了解该实例化对象帮助不大。
        事实上，当我们输出某个实例化对象时，其调用的就是该对象的 __repr__() 方法，输出的是该方法的返回值。
        通过重写类的 __repr__() 方法可以自定义输出实例化对象时的信息。"""
        return self._data_dict.__repr__()

    def to(self, device):
        for key, value in self._data_dict.items():
            self._data_dict[key] = value.to(device)
        return self


if __name__ == '__main__':
    container = Container([1, 2, 3], x=1, y=2)

    # test __setattr__
    print("test __setattr__()")
    container.xxx = 3
    print(container.__dict__)
    print()

    # test __getitem__
    print("test __getitem()__")
    print(container['x'])
    print()

    # test __iter__
    print("test __iter__()")
    c_iter = iter(container)
    print(next(c_iter))
    print(next(c_iter))
    print()

    # test __setitem__
    print("test __setitem__()")
    container['z'] = 5
    print(container)
    print('--------------')

    for key, value in container:
        print(key)
