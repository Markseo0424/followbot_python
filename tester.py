import numpy as np

class Foo:
    def __init__(self, elements: list):
        self.element = elements

    def __len__(self):
        return len(self.element)

    def __getitem__(self, item):
        return self.element[item]


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    foo = Foo(a)
    print(a[0])
    print(foo[0])
    print(a[0:3])
    print(foo[0:3])
    print(a[[0,2,4]])
    print(foo[[0,2,4]])

