# 0 is COMPLETE LINK, 1 is SINGLE
def cluster(distances, link):
    clusters = {}
    for idx in xrange(1,5):
        clusters[(idx,)] = (idx,)
    # print clusters
    counter = 0
    while len(clusters) > 1:
        nearest_pair = (None, None)
        nearest_dist = 1000
        counter += 1
        # Go through every cluster-pair to find nearest pair
        for a in clusters:
            for b in clusters:
                # Do not compare with same cluster
                if a is not b:
                    if link is 0:
                        dist = link
                    elif link is 1:
                        dist = 1000
                    # For each element in both clusters, determine "nearness"
                    # Complete nearness: farthest distance between any two elements in two clusters
                    # Single nearness: the nearest distance between any two elemeents in two clusters
                    for i in clusters[a]:
                        for j in clusters[b]:
                            if i < j:
                                k = (i,j)
                            elif i > j:
                                k = (j,i)
                            else:
                                continue
                            curr_dist = distances[k]
                            if (link is 0 and curr_dist > dist) or (link is 1 and curr_dist < dist):
                                print counter, ': New distance for', clusters[a], clusters[b], dist, '->', curr_dist
                                dist = curr_dist
                    # Find out if this is the nearest pair so far in the iteration
                    if dist < nearest_dist:
                        print counter, ': Replace distance with ', (a,b), nearest_dist, '->', dist
                        nearest_dist = dist
                        nearest_pair = (a,b)
                        # add elements in a tuple
                        nearest_pair_values = clusters[a]+clusters[b]
        clusters.pop(nearest_pair[0])
        clusters.pop(nearest_pair[1])
        # new_pair = nearest_pair[0] + nearest_pair[1]
        clusters[nearest_pair_values] = nearest_pair_values
        # clusters[new_pair] = clusters[new_pair]
        print counter, clusters.keys()
    # print clusters
    return clusters.keys()

dic = {(1,1): 0, (1,2): 0.5, (1,3): 0.1, (1,4): 0.2, (2,3): 0.4, (2,4): 0.6, (3,4): 0.3}
complete = cluster(dic, 0)
print complete
