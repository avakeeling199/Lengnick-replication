from agents import Household, Firm
from model import LegnickModel
import matplotlib.pyplot as plt

model = LegnickModel()
for i in range(20000):
    model.step()

data = model.datacollector.get_model_vars_dataframe()
data.to_csv('diagnostics/diagnostic_run_20k.csv')

fig, axes = plt.subplots(6, 1, figsize=(10, 10))
axes[0].plot(data['Employment'])
axes[0].set_title('Employment')
axes[1].plot(data['AvgPrice'])
axes[1].set_title('Average Price')
axes[2].plot(data['AvgWage'])
axes[2].set_title('Average Wage')
axes[3].plot(data['TotalInv'])
axes[3].set_title('Total Inventory')
axes[4].plot(data['PositionsStd'])
axes[4].set_title('Positions Std')
axes[5].plot(data['AvgPositions'])
axes[5].set_title('Average Positions')
plt.tight_layout()
plt.savefig('diagnostics/diagnostic_run_20k.png', dpi=150)