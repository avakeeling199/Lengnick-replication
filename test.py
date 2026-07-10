from agents import Household, Firm
from model import LegnickModel
import matplotlib.pyplot as plt

model = LegnickModel(seed=42)
for i in range(31500):
    model.step()

data = model.datacollector.get_model_vars_dataframe()
data.to_csv('diagnostics/diagnostic_run_20k.csv')

fig, axes = plt.subplots(4, 1, figsize=(10, 10))
axes[0].plot(data['Employment'])
axes[0].set_title('Employment')
axes[1].plot(data['AvgPrice'])
axes[1].set_title('Average Price')
axes[2].plot(data['AvgWage'])
axes[2].set_title('Average Wage')
axes[3].plot(data['TotalInv'])
axes[3].set_title('Total Inventory')

plt.tight_layout()
plt.savefig('diagnostics/diagnostic_run_20k.png', dpi=150)