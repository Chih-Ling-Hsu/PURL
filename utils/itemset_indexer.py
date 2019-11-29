
class ItemSetIndexer():
    def __init__(self):
        self.indexTable = None
        self.itemsetTable = None
        self.size = 0
#         self.indexTable = whiteList
#         self.itemsetTable = dict((v,i) for (i,v) in enumerate(self.indexTable))
#         self.size = len(whiteList)
    
    '''
    @Param X: (List) itemsets
    '''    
    def fit(self, X):
        self.indexTable = list(set(X))
        self.itemsetTable = dict((v,i) for (i,v) in enumerate(self.indexTable))
        self.size = len(self.indexTable)
        return self
        
#     '''
#     @Param X: (List) itemsets
#     '''    
#     def partial_fit(self, X):        
#         #_filter = [x not in self.indexSet for x in X]
#         #addList = list(itertools.compress(X, _filter))
#         addSet = set(filter(lambda x: x not in self.indexSet, X)) & self.whiteList
        
#         self.indexSet = self.indexSet | addSet
#         self.indexTable += list(addSet)
#         self.size += len(addSet)
    
    def transform(self, x):  
        if x in self.itemsetTable:
            return self.itemsetTable[x]
        else:
            return -1
    
    def inv_transform(self, y):    
        return self.indexTable[int(y)]
    
    '''
    @Param X: (List) exercise patterns
    @Return : (List) pattern codes
    '''
    def encode(self, X):
        codes = map(self.transform, X)
        codes = filter(lambda x: x != -1, codes)
        
        #if len(codes) == 0:
        #    codes = [self.size]
            
        return sorted(codes)
    
    '''
    @Param Y: (List) pattern codes
    @Return : (List) exercise patterns
    '''
    def decode(self, Y):
        patterns = map(self.inv_transform, Y)
        return patterns