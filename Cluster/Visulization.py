import matplotlib.pyplot as plt

def clusterClubs(clusters):
    fig, ax = plt.subplots()
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']

    for i, v in enumerate(clusters):
        print(clusters[v]['data'].shape)
        ax.scatter(
            # xRange, yRange,
            clusters[v]['data'][:, 0], clusters[v]['data'][:, 1],
            marker=scatterMarkers[i]
        )
        ax.scatter(
            # xRange, yRange,
            clusters[v]['mean_vec'][0], clusters[v]['mean_vec'][1],
            marker='+',
            s=600,
        )
    plt.show()
