import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    'Category': [],
    'Values': []
}

MAPPING = {
    "no partition": "no par",
    "1g.5gb isolated": "1g.5gb",
    # "1g.5gb+me isolated": "1g.5gb+me",
    "1g.10gb isolated": "1g.10gb",
    "2g isolated": "2g",
    "3g isolated": "3g",
    "4g isolated": "4g",
    "7g isolated": "7g",
}

def read_bench(fname):
    f = open(fname)
    lines = [l.rstrip() for l in f.readlines()]
    for row, line in enumerate(lines):
        s = line.split(',')
        if s[0] in MAPPING:
            data['Category'].append(MAPPING[s[0]])
            data['Values'].append(float(s[1]) * 1000.0)
    f.close()
    
read_bench('/users/pzhou/projects/gpuless/benchmark-results/mig-isolation/mig-results/mig-isolation-bench-2024-03-27T22:37:52+01:00.out')

# Convert data to a DataFrame
import pandas as pd
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

# Create a violin plot
sns.violinplot(x='Category', y='Values', data=df)

# Add title and labels
# plt.title('Runtime of warm ResNet-50 benchmark for 7 MIG partition profiles on A100, n=100.')
plt.title('Runtime of warm ResNet-50 benchmark for 6 MIG partition profiles on A100, n=100.')
plt.xlabel('Partition')
plt.ylabel('Milliseconds(ms)')

plt.xticks(rotation=90)

# save the plot
plt.savefig('violin_plot.pdf', bbox_inches='tight')
# plt.savefig('violin_plot_bfs.pdf', bbox_inches='tight')
# plt.savefig('violin_plot_bfs.pdf', bbox_inches='tight')
