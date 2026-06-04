
def func():
    print("方法被执行了!!")
    yield 1

func()
print(type(func()))

# for item in func():
#     pass