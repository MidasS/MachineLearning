class foo():
    def __init__(self,date):
        self._date2 = date

    @property
    def getx(self):
        # print(self.date2)
        return self._date2


a = foo(3)
# a._date2 = 5
print(a.getx)

