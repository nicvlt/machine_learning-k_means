import pandas as pd
import numpy as np
import sklearn.datasets as skDatasets
import sklearn.cluster as skCluster
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def initialize_csv():
    X = pd.read_csv(r'C:/Users/nicol/Desktop/Cours/L3/Machine Learning/TP2/assets/clustering.csv')
    df_input = pd.DataFrame(X, columns=['ApplicantIncome', 'LoanAmount'])
    df_input['Cluster'] = ''
    return df_input

def get_euclidean_distance(pointM, tab):
    #pointM is a 2d point, calculate the euclidean distance between pointM and all the points in tab
    distances = np.array([])
    for key in tab:
        euclidean_distance = np.sqrt((pointM[0]- tab[key][0])**2 + (pointM[1]- tab[key][1])**2)
        distances = np.append(distances, euclidean_distance)
    return distances

def get_euclidean_distance_points_to_point(pointX, pointY):
    return np.sqrt((pointX[0]- pointY[0])**2 + (pointX[1]- pointY[1])**2)

def get_gravcenter_centroids(centroids):
    #get the gravcenter of the centroids
    sumX = 0
    sumY = 0
    for key in centroids:
        sumX += centroids[key][0]
        sumY += centroids[key][1]
    return [sumX/len(centroids), sumY/len(centroids)]

def init_centroids(K, df):
    #randomly choose K points from df
    centroids = {}
    points_exist = np.array([])
    k = 1
    while(k < K + 1):
        random = np.random.randint(1, df.shape[0])
        if(random in points_exist):
            continue
        points_exist = np.append(points_exist, random)
        centroids['{}'.format(k)] = [df[df.columns[0]][random], df[df.columns[1]][random]]
        k += 1
    return centroids

def assign_centroids(df, centroids):
    Ydict= {}
    for i in range(0, df.shape[0]):
        distances = get_euclidean_distance([df[df.columns[0]][i], df[df.columns[1]][i]], centroids)
        Ydict['{}'.format(i)] = np.where(distances == np.amin(distances))[0][0]
    return Ydict
    
def updated_centroids(df, centroids):
    points_to_clusters = assign_centroids(df, centroids)
    df['Cluster'] = points_to_clusters.values()

def compute_interia_within(df, centroid):
    #sum of the euclidean distance between each point and the centroid
    sum = 0
    for index in df.index:
        sum += get_euclidean_distance_points_to_point([df[df.columns[0]][index], df[df.columns[1]][index]], centroid)
    return sum

def computer_inertia_between(centroids, gravcenter):
    sum = 0
    for key in centroids:
        sum += get_euclidean_distance_points_to_point(centroids[key], gravcenter)
    return sum

def compute_inertiaW(dfB, centroids):
    W = 0
    B = 0
    for key in centroids:
        df = dfB[dfB['Cluster'] == int(key) - 1]
        W += compute_interia_within(df, centroids[key])
        B += computer_inertia_between(centroids, get_gravcenter_centroids(centroids))
    return W + B

def new_centroids(dfB, centroids):
    #calculate new centroids
    temp_centroids = centroids.copy()
    for key in centroids:
        df = dfB[dfB['Cluster'] == int(key) - 1]
        centroids[key] = [df[df.columns[0]].mean(), df[df.columns[1]].mean()]
    assign_centroids(dfB, centroids)
    updated_centroids(dfB, centroids)
    if(temp_centroids == centroids):
        return 1
    return 0

def add_to_plot(centroids, K):
    for i in range(1, K+1):
        plt.scatter(centroids['{}'.format(i)][0], centroids['{}'.format(i)][1], c='red', marker='X', alpha=0.3)

def plot_k_means(df, K, centroids):
    #plot each points with their cluster and the centroids of each cluster with X
    plt.scatter(df[df.columns[0]], df[df.columns[1]], c=df['Cluster'])
    for i in range(1, K+1):
        plt.scatter(centroids['{}'.format(i)][0], centroids['{}'.format(i)][1], c='blue', marker='X', s=500, alpha=0.6)
    plt.show()


def K_means(df, K):
    #calculate new centroids
    centroids = init_centroids(K, df)

    #assign new centroids to each point
    updated_centroids(df, centroids)
    
    #get intraclass inertia
    compute_inertiaW(df, centroids)

    i=0
    loop = True
    while(loop):
        check_loop = new_centroids(df, centroids)
        add_to_plot(centroids, K)
        print('New step ', i+1)
        i += 1
        if(check_loop == 1):
            loop = False

    #plot_k_means(df, K, centroids)
    plot_k_means(df, K, centroids)


def main():

    #initialise the dataset
    df = initialize_csv()
    K=3

    print("Choose an action :\n")
    usr_input = int(input("1. K-means algorithm\n2. K-means algorithm with scikit-learn\n3. Exit\n"))

    if(usr_input == 1):
        K_means(df, K)

    elif(usr_input == 2):
        pointSet = skDatasets.make_blobs(n_samples=100, n_features = 2, centers=K)
        temp = {'X' : pointSet[0][:,0], 'Y' : pointSet[0][:,1], 'Cluster' : pointSet[1]}
        dfB = pd.DataFrame(temp, columns=['X', 'Y', 'Cluster'])

        KMeans = skCluster.KMeans(n_clusters = K).fit(dfB[['X', 'Y']])

        plt.scatter(dfB['X'], dfB['Y'], c=dfB['Cluster'])
        plt.scatter(KMeans.cluster_centers_[:,0], KMeans.cluster_centers_[:,1], c='red', marker='X', s=500, alpha=0.6)
        plt.show()
    
    elif(usr_input == 3):
        exit()

main()