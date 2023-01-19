import math
import random

from matplotlib import pyplot as plt

countPoints = 15
countClusters = 4
random.seed()


def init_population():
    return [[random.randint(1, 15), random.randint(1, 15)] for _ in range(countPoints)]


def init_centers_clusters():
    return [[random.randint(1, 15), random.randint(1, 15)] for _ in range(countClusters)]


def init_clusters():
    return [[] for i in range(countClusters)]


def init_healths_chromosomes():
    return [[] for i in range(countClusters)]


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def squared_errors(distances):
    return sum([distanсe ** 2 for distanсe in distances])


def squared_errors_matrix(distances):
    li = []
    for distance in distances:
        li += distance
    return sum(li)


def points_in_clusters(points, centers, clusters, min_distances_points):
    for point in points:
        in_cluster = [euclidean_distance(point, center) for center in centers]
        clusters[in_cluster.index(min(in_cluster))].append(point)
        min_distances_points[in_cluster.index(min(in_cluster))].append(min(in_cluster))


def cross(chromosome1, chromosome2):
    length = 4
    l = random.randint(1, length)

    str1 = bin(chromosome1)
    str1 = str1[2:]
    str1 = list(str1)
    str1[0] = '0' * (length - len(str1)-1) + str1[0]
    str1 = ''.join(str1)

    str2 = bin(chromosome2)
    str2 = str2[2:]
    str2 = list(str2)
    str2[0] = '0' * (length - len(str2)-1) + str2[0]
    str2 = ''.join(str2)

    tmpx1 = str1[:l]
    tmpy1 = str1[l:]
    tmpx2 = str2[:l]
    tmpy2 = str2[l:]

    new_chromosome = tmpx1 + tmpy2
    new_chromosome1 = tmpx2 + tmpy1
    return int(new_chromosome, 2), int(new_chromosome1, 2)


def mutation(chromosome):
    strtmp = bin(chromosome)
    strtmp = strtmp[2:]
    strtmp = list(strtmp)
    l = random.randint(0, len(strtmp) - 1)
    strtmp[l] = str((int(strtmp[l]) + 1) % 2)
    strtmp = ''.join(strtmp)
    strtmp = int(strtmp, 2)
    chromosome = strtmp


def selection_chromosomes(min_distances_points, centers):
    parents_pool = []
    healths = [squared_errors(distances) for distances in min_distances_points]
    sum_healths = sum(healths)
    probabilities = [(health / sum_healths) for health in healths]
    # prod_big = sorted(probabilities)
    # prob_less = sorted(probabilities, reverse=True)
    # for i in range(2):
    #     indexless = probabilities.index(prob_less[i])
    #     indexbig = probabilities.index(prod_big[i])
    #     t = probabilities[indexless]
    #     probabilities[indexless] = probabilities[indexbig]
    #     probabilities[indexbig] = t
    roulette = [random.randint(0, 99) for _ in range(countClusters)]
    for number in roulette:
        i = 1
        while number / 100 >= sum(probabilities[:i]):
            i += 1
        parents_pool.append(centers[i - 1])
    return parents_pool


pop = [[13, 10], [9, 7], [10, 10], [15, 7], [9, 15], [4, 9], [2, 2], [8, 1], [13, 4], [2, 6], [11, 9], [9, 14], [7, 11],
       [13, 4], [4, 14]]
clusters = init_clusters()
centers = init_centers_clusters()
min_distances_points = init_healths_chromosomes()
points_in_clusters(pop, centers, clusters, min_distances_points)
best = squared_errors_matrix(min_distances_points)
best_centers = centers
best_clusters = clusters
best_min_distances_points = min_distances_points
i = 0
print(best)
print(best_centers)
print(best_clusters)
while (i<10000):
    parents_pool = selection_chromosomes(min_distances_points, centers)
    p = random.random()
    if p < 0.1:
        mutation(parents_pool[random.randint(0, countClusters - 1)][random.randint(0, 1)])
    else:
        parents_pool[0] = cross(parents_pool[0][0],
                                parents_pool[1][0])
        parents_pool[1] = cross(parents_pool[0][1],
                                parents_pool[1][1])
        parents_pool[2] = cross(parents_pool[2][0],
                                parents_pool[3][0])
        parents_pool[3] = cross(parents_pool[2][1],
                                parents_pool[3][1])
    centers = parents_pool
    min_distances_points = init_healths_chromosomes()
    clusters = init_clusters()
    points_in_clusters(pop, centers, clusters, min_distances_points)
    if best >= squared_errors_matrix(min_distances_points):
        best = squared_errors_matrix(min_distances_points)
        best_centers = centers
        best_clusters = clusters
        best_min_distances_points = min_distances_points
    else:
        centers = best_centers
        min_distances_points = best_min_distances_points
    i += 1
print(best)
print(best_centers)
print(best_clusters)
plt.scatter([x[0] for x in best_centers],[y[1] for y in best_centers],marker='*',linewidths=10)
for cluster in best_clusters:
    plt.scatter([x[0] for x in cluster],[y[1] for y in cluster])
plt.show()
