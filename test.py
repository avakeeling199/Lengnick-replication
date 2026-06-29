from agents import Household, Firm
from model import LegnickModel

model = LegnickModel()
for i in range(100):
    model.step()
print("ran ok")