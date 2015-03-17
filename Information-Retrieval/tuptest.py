def cluster_id(list_of_tups, key):
    cluster_ids = [0] * 40
    for i in xrange(7):
        tup = list_of_tups[i]
        for j in tup:
            if key is 1:
                cluster_ids[j-1] = i
            else:
                cluster_ids[j] = i
    return cluster_ids

ashley_c = \
[(1,3,4,8,10,16),
(11,15,5,6,23,40,20,33,7),
(34,2),
(28,18,21,39,26,17,35,25),
(12,9,13,14),
(27,32,31),
(37,19,22,36,29,24,38,30)]

print cluster_id(ashley_c, 1)